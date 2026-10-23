"""Outside grokking: do training margins of an ordinary classifier ever reach the fp32 cliff?

sklearn's bundled 8x8 digits (1,797 images, no download), 80/20 split, MLP 64-256-256-10,
full-batch AdamW lr 1e-3, eps 1e-8, 20,000 steps, fp32. Standard vs fixed CE, wd in {0, 0.01}.
usage: python digits.py LOSS WD SEED -> results/digits/LOSS_wdWD_sSEED.json
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from ce import fixed_ce, margin

STEPS = int(os.environ.get("STEPS", 20_000))


def main(loss_kind, wd, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X / 16.0, dtype=torch.float32); y = torch.tensor(y)
    perm = torch.randperm(len(X)); n = int(0.8 * len(X))
    tr, te = perm[:n], perm[n:]
    net = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    log = []
    for step in range(STEPS + 1):
        z = net(X[tr])
        loss = F.cross_entropy(z, y[tr]) if loss_kind == "std" else fixed_ce(z, y[tr])
        if step % 50 == 0:
            with torch.no_grad():
                m = margin(z, y[tr])
                q = torch.softmax(z, -1).gather(-1, y[tr][:, None]).squeeze(-1)
                te_acc = (net(X[te]).argmax(-1) == y[te]).float().mean().item()
            log.append([step, F.softplus(-m).mean().item(), m.median().item(), m.min().item(),
                        (q == 1.0).float().mean().item(), te_acc])
        if step == STEPS:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    os.makedirs("results/digits", exist_ok=True)
    json.dump(dict(loss=loss_kind, wd=wd, seed=seed, cols=["step", "exact_train_ce", "median_margin", "min_margin",
              "frac_target_grad_zero", "test_acc"], log=log, torch=torch.__version__),
              open(f"results/digits/{loss_kind}_wd{wd}_s{seed}.json", "w"))
    print(loss_kind, wd, seed, log[-1], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
