"""Pairwise vs pointwise age regression on UTKFace at small N.
Arms (same backbone, optimizer, batch size, steps, augmentation):
  pair   : 2-channel pair input, CE on "which is older", entropy decode vs references (paper)
  mse    : 1-channel input, MSE on standardized age
  l1     : 1-channel input, L1 on standardized age
  mse_raw: 1-channel input, MSE on raw age in years (no target standardization)
Usage: python3 exp.py --arm pair --n 100 --seed 0 [--sampling balanced|natural]
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- backbone: the paper's AgeRanker (ResNet-10 style), copied structurally ----
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(out + (x if self.down is None else self.down(x)))


class Net(nn.Module):
    def __init__(self, in_ch, out_dim, base=32, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.body = nn.Sequential(ResBlock(base, base), ResBlock(base, 2 * base, 2),
                                  ResBlock(2 * base, 4 * base, 2), ResBlock(4 * base, 8 * base, 2))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(8 * base, out_dim)

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.fc(self.drop(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)))


# ---- data / splits ----------------------------------------------------------
def load():
    d = np.load(os.path.join(HERE, "data", "utk64.npz"))
    x = torch.from_numpy(d["x"]).float().div(255).sub(0.5).div(0.5).unsqueeze(1)  # [-1,1], as in the paper
    return x, d["age"].astype(np.float32)


def splits(ages, n, seed, n_test=200, sampling="balanced"):
    """Test: n_test images at evenly spaced positions of the age-sorted pool (fixed across seeds).
    Train: paper's decade-balanced sampling (n//10 per decade, with replacement if short), or natural."""
    order = np.argsort(ages, kind="stable")
    test = order[np.linspace(0, len(order) - 1, n_test).round().astype(int)]
    pool = np.setdiff1d(np.arange(len(ages)), test)
    rng = np.random.default_rng(seed)
    if sampling == "natural":
        return rng.choice(pool, n, replace=False), test
    train = []
    for dec in range(10):
        b = pool[np.minimum(ages[pool] // 10, 9) == dec]
        k = n // 10
        train.extend(rng.choice(b, k, replace=len(b) < k))
    return np.array(train), test


def flip(x, p=0.5):
    m = torch.rand(x.shape[0]) < p
    x = x.clone(); x[m] = x[m].flip(-1)
    return x


# ---- decoding ----------------------------------------------------------------
@torch.no_grad()
def pair_decode(model, x, ages, test, refs, bs=1024):
    """Paper's entropy decode: per reference age, mean P(test older); smooth (sigma=1 over the
    sorted-age index); predict the age with max binary entropy."""
    model.eval()
    ref_ages = ages[refs]
    uniq = np.unique(ref_ages)
    xr = x[refs]
    preds, votes_all = [], []
    for t in test:
        xt = x[t:t + 1].expand(len(refs), -1, -1, -1)
        p_older = torch.cat([model(torch.cat([xt[k:k + bs], xr[k:k + bs]], 1)).softmax(1)[:, 0]
                             for k in range(0, len(refs), bs)]).numpy()
        votes = np.array([p_older[ref_ages == a].mean() for a in uniq])
        v = np.clip(gaussian_filter1d(votes, 1.0), 1e-6, 1 - 1e-6)
        ent = -(v * np.log(v) + (1 - v) * np.log(1 - v))
        preds.append(uniq[np.argmax(ent)])
    return np.array(preds, dtype=np.float32)


@torch.no_grad()
def point_predict(model, x, idx, mu, sd):
    model.eval()
    return (model(x[idx]).squeeze(1) * sd + mu).numpy()


# ---- training -----------------------------------------------------------------
def run(args):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    x, ages = load()
    train, test = splits(ages, args.n, args.seed, sampling=args.sampling)
    y = torch.from_numpy(ages)
    mu, sd = float(ages[train].mean()), float(ages[train].std())
    if args.arm == "mse_raw":
        mu, sd = 0.0, 1.0  # the head has to output years directly
    model = Net(2 if args.arm == "pair" else 1, 2 if args.arm == "pair" else 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    g = torch.Generator().manual_seed(args.seed)
    curve, t0 = [], time.time()
    tr = torch.from_numpy(train)

    def evaluate():
        if args.arm == "pair":
            pred = pair_decode(model, x, ages, test, train)
        else:
            pred = point_predict(model, x, test, mu, sd)
        return pred, float(np.abs(pred - ages[test]).mean())

    for step in range(1, args.steps + 1):
        model.train()
        i = tr[torch.randint(len(tr), (args.batch,), generator=g)]
        if args.arm == "pair":
            j = tr[torch.randint(len(tr), (args.batch,), generator=g)]
            keep = i != j
            i, j = i[keep], j[keep]
            xb = torch.cat([flip(x[i]), flip(x[j])], 1)
            lab = (y[i] <= y[j]).long()  # paper: 0 if age_i > age_j else 1
            loss = F.cross_entropy(model(xb), lab)
        else:
            out = model(flip(x[i])).squeeze(1)
            tgt = (y[i] - mu) / sd
            loss = F.l1_loss(out, tgt) if args.arm == "l1" else F.mse_loss(out, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            pred, mae = evaluate()
            curve.append({"step": step, "loss": float(loss), "test_mae": mae, "t": time.time() - t0})
            print(json.dumps(curve[-1]), flush=True)
    pred, mae = evaluate()
    res = {"args": vars(args), "train_mean": mu, "train_sd": sd, "final_test_mae": mae,
           "best_on_test_mae": min(c["test_mae"] for c in curve), "curve": curve,
           "pred": pred.tolist(), "true": ages[test].tolist(), "wall_s": time.time() - t0}
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, f"{args.arm}_{args.sampling}_n{args.n}_s{args.seed}.json"), "w"))
    print("FINAL", args.arm, args.n, args.seed, "mae %.2f best-on-test %.2f pred sd %.1f wall %.0fs"
          % (mae, res["best_on_test_mae"], pred.std(), res["wall_s"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["pair", "mse", "l1", "mse_raw"], required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling", choices=["balanced", "natural"], default="balanced")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    run(a)
