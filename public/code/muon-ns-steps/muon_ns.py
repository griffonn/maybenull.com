"""Muon with different orthogonalizers on a tiny char-level GPT (CPU).

Arms: Newton-Schulz with q steps (Muon's quintic coefficients), exact polar
via SVD, damped polar  U diag(s/sqrt(s^2+eps)) V^T  with eps = l_q^-2, and AdamW.
"""
import argparse, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F

A, B, C = 3.4445, -4.7750, 2.0315  # Muon quintic coefficients (Jordan et al.)


def frob_normalize(G):
    return G / (G.norm() + 1e-7)


def ns_quintic(G, q):
    X = frob_normalize(G)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    for _ in range(q):
        S = X @ X.T
        X = A * X + (B * S + C * S @ S) @ X
    return X.T if tr else X


def polar(G, eps=0.0):
    X = frob_normalize(G)
    U, s, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    f = torch.ones_like(s) if eps == 0 else s / torch.sqrt(s * s + eps)
    return ((U * f) @ Vh).float()


def make_ortho(name):
    if name.startswith("ns"):
        q = int(name[2:]); return lambda G: ns_quintic(G, q)
    if name == "polar":
        return lambda G: polar(G)
    if name.startswith("damped"):  # damped<q>: eps = l_q^-2, l_q = A^q
        q = int(name[6:]); eps = A ** (-2 * q); return lambda G: polar(G, eps)
    raise ValueError(name)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, ortho, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.ortho = ortho

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"])  # Nesterov
                O = self.ortho(upd)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(O, alpha=-g["lr"] * scale)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        Bt, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, 2)
        q, k, v = (t.view(Bt, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(Bt, T, D))
        return x + self.fc2(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, V, d, L, h, T):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lr", type=float, required=True)  # Muon lr (AdamW arm: adam lr)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--spectra", action="store_true")
    ap.add_argument("--out", default="results/runs")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    torch.manual_seed(a.seed)
    text = open(os.path.join(os.path.dirname(__file__), "data/input.txt")).read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    n = int(0.9 * len(data)); tr, va = data[:n], data[n:]
    g = torch.Generator().manual_seed(1234)  # fixed val batches
    vidx = torch.randint(len(va) - a.T - 1, (8, 32), generator=g)

    def batch(src, ix):
        return (torch.stack([src[i:i + a.T] for i in ix]), torch.stack([src[i + 1:i + a.T + 1] for i in ix]))

    model = GPT(len(chars), a.d, a.L, 4, a.T)
    hidden = [p for n_, p in model.named_parameters() if p.ndim == 2 and "blocks" in n_]
    other = [p for n_, p in model.named_parameters() if not (p.ndim == 2 and "blocks" in n_)]
    if a.arm == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)]
    else:
        opts = [Muon(hidden, a.lr, make_ortho(a.arm)),
                torch.optim.AdamW(other, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)]
    dg = torch.Generator().manual_seed(a.seed)
    log = []

    @torch.no_grad()
    def evaluate():
        model.eval(); tot = 0.0
        for row in vidx:
            x, y = batch(va, row)
            tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
        model.train(); return tot / len(vidx)

    t0 = time.time(); opt_time = 0.0
    for step in range(a.steps + 1):
        lr_mult = min(1.0, (a.steps - step) / (0.3 * a.steps))  # linear decay over last 30%
        for o in opts:
            for gr in o.param_groups:
                gr.setdefault("base_lr", gr["lr"]); gr["lr"] = gr["base_lr"] * lr_mult
        if step % a.eval_every == 0 or step == a.steps:
            vl = evaluate(); log.append(dict(step=step, val=vl, time=time.time() - t0))
            if not math.isfinite(vl):
                break
        if step == a.steps:
            break
        x, y = batch(tr, torch.randint(len(tr) - a.T - 1, (a.bs,), generator=dg))
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        ts = time.time()
        for o in opts:
            o.step()
        opt_time += time.time() - ts
    res = dict(vars(a), log=log, final_val=log[-1]["val"], wall=time.time() - t0, opt_time=opt_time,
               n_params=sum(p.numel() for p in model.parameters()))
    if a.spectra and a.arm != "adamw":
        sp = {}
        mu = opts[0]
        names = {id(p): n_ for n_, p in model.named_parameters()}
        for p in hidden:
            s = torch.linalg.svdvals(frob_normalize(mu.state[p]["buf"]).double())
            sp[names[id(p)]] = s.tolist()
        res["momentum_spectra"] = sp
    os.makedirs(a.out, exist_ok=True)
    fn = f"{a.out}/{a.arm}_lr{a.lr:g}_s{a.seed}_n{a.steps}.json"
    json.dump(res, open(fn, "w"))
    print(f"{a.arm} lr={a.lr:g} seed={a.seed} final_val={res['final_val']:.4f} wall={res['wall']:.1f}s opt={opt_time:.1f}s")


if __name__ == "__main__":
    main()
