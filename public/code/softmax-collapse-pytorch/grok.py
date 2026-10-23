"""Grokking a+b mod 31 with standard vs fixed cross-entropy, and cheap knobs.

Setup follows the paper's Sec. 6 description (arXiv 2610.31642): 1-layer width-64 ReLU transformer,
50% train split, full-batch AdamW, betas (0.9, 0.98), weight decay 1.0, 10,000 steps.
We use lr 3e-3 (the paper uses 1e-3; at 1e-3 our model had not grokked by step 10k in pilots).
Our transformer: token+learned position embeddings, 4-head attention, ReLU MLP (256), residuals,
no LayerNorm, unembedding read at the '=' position.

usage: python grok.py ARM SEED   -> results/grok/ARM_sSEED.json
"""
import json, math, os, sys, time, platform
import torch, torch.nn as nn, torch.nn.functional as F
from ce import fixed_ce, margin

ARMS = {
    # name: (loss, param dtype, autocast, adam eps, label smoothing)
    "std_fp32":       ("std",    torch.float32,  False, 1e-8, 0.0),
    "fixed_fp32":     ("fixed",  torch.float32,  False, 1e-8, 0.0),
    "std_fp32_eps1e-7": ("std",  torch.float32,  False, 1e-7, 0.0),
    "std_fp32_ls0.01": ("std",   torch.float32,  False, 1e-8, 0.01),
    "std_fp32_loss64": ("std64", torch.float32,  False, 1e-8, 0.0),
    "std_bf16":       ("std",    torch.bfloat16, False, 1e-8, 0.0),
    "fixed_bf16":     ("fixed",  torch.bfloat16, False, 1e-8, 0.0),
    "std_autocast_bf16": ("std", torch.float32,  True,  1e-8, 0.0),
    # fp32 weights + autocast forward, but the loss is computed on the bf16 logits (autocast off):
    # isolates the bf16 cliff from bf16 weight-update rounding.
    "std_bf16logits":   ("std_bf16z",   torch.float32, True, 1e-8, 0.0),
    "fixed_bf16logits": ("fixed_bf16z", torch.float32, True, 1e-8, 0.0),
}
P, STEPS, LOG_EVERY = 31, int(os.environ.get("STEPS", 10_000)), 10
D = int(os.environ.get("D", 64))
TOK4 = os.environ.get("TOK4", "0") == "1"   # <a><+><b><=> input as in Varma et al.
LR = float(os.environ.get("LR", 3e-3))  # paper: 1e-3; our model does not grok within 10k steps at 1e-3 (see README)
NT = 4 if TOK4 else 3


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(P + 2, D)
        self.pos = nn.Parameter(torch.randn(NT, D) * 0.02)
        self.att = nn.MultiheadAttention(D, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))
        self.out = nn.Linear(D, P)

    def forward(self, x):
        h = self.emb(x) + self.pos
        h = h + self.att(h, h, h, need_weights=False)[0]
        h = h + self.mlp(h)
        return self.out(h[:, -1])


def main(arm, seed):
    loss_kind, pdtype, autocast, eps, ls = ARMS[arm]
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    a, b = a.flatten(), b.flatten()
    X = (torch.stack([a, torch.full_like(a, P + 1), b, torch.full_like(a, P)], 1) if TOK4
         else torch.stack([a, b, torch.full_like(a, P)], 1))
    Y = (a + b) % P
    perm = torch.randperm(len(X))
    tr, te = perm[: len(X) // 2], perm[len(X) // 2:]
    net = Net().to(pdtype)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.98), eps=eps, weight_decay=1.0)
    log = []
    t0 = time.time()
    for step in range(STEPS + 1):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            z = net(X[tr])
            if loss_kind == "std":
                loss = F.cross_entropy(z, Y[tr], label_smoothing=ls)
            elif loss_kind == "std64":
                loss = F.cross_entropy(z.double(), Y[tr])
            elif loss_kind == "std_bf16z":
                with torch.autocast("cpu", enabled=False):
                    loss = F.cross_entropy(z.bfloat16(), Y[tr])
            elif loss_kind == "fixed_bf16z":
                with torch.autocast("cpu", enabled=False):
                    loss = fixed_ce(z.bfloat16(), Y[tr])
            else:
                loss = fixed_ce(z, Y[tr])
        if step % LOG_EVERY == 0:
            with torch.no_grad():
                zd = z.detach()
                m = margin(zd, Y[tr])
                exact = F.softplus(-m).mean().item()          # true CE of these logits, fp64
                # fraction of train examples whose standard-CE target gradient is exactly 0 in the logits' dtype
                q = torch.softmax(zd.float() if zd.dtype == torch.float32 else zd, -1)
                dead = (q.gather(-1, Y[tr][:, None]).squeeze(-1) == 1.0).float().mean().item()
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                    zt = net(X[te])
                te_acc = (zt.argmax(-1) == Y[te]).float().mean().item()
                tr_acc = (zd.argmax(-1) == Y[tr]).float().mean().item()
            log.append([step, exact, float(loss.item()), m.mean().item(), m.min().item(), dead, tr_acc, te_acc])
        if step == STEPS:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    out = dict(arm=arm, seed=seed, cfg=dict(loss=loss_kind, dtype=str(pdtype), autocast=autocast, eps=eps,
               label_smoothing=ls, P=P, D=D, steps=STEPS, lr=LR, betas=[0.9, 0.98], wd=1.0, split=0.5),
               cols=["step", "exact_train_ce", "loss_value", "mean_margin", "min_margin",
                     "frac_target_grad_zero", "train_acc", "test_acc"],
               log=log, seconds=time.time() - t0, torch=torch.__version__, python=platform.python_version())
    os.makedirs("results/grok", exist_ok=True)
    with open(os.environ.get("OUT", f"results/grok/{arm}_s{seed}.json"), "w") as f:
        json.dump(out, f)
    print(f"{arm} s{seed}: test {log[-1][7]:.3f} margin {log[-1][3]:.2f} {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
