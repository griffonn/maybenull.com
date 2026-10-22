"""Tiny from-scratch looped transformer on swap-S5, testing the PAR recipe
(arXiv 2610.27628): progress-aligned targets x RMS norm anchoring.

Usage: python3 par.py --targets progress|final --norm 1|0 --seed 0 --out results/
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- task: swap-S5 -------------------------------------------------------
# 5 boxes hold items 0..4 (box i holds item i at the start). A move is the
# identity (token 0) or one of the 10 transpositions (tokens 1..10).
SWAPS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
N_MOVES, QRY = 11, 11
SWAP_A = torch.tensor([0] + [i for i, _ in SWAPS])  # identity swaps box 0 with itself
SWAP_B = torch.tensor([0] + [j for _, j in SWAPS])


def make_batch(n, d, gen):
    """Returns tokens [n, d+1] and states [n, d+1, 5] (state after 0..d moves)."""
    moves = torch.randint(0, N_MOVES, (n, d), generator=gen)
    state = torch.arange(5).repeat(n, 1)
    states = [state.clone()]
    rows = torch.arange(n)
    for t in range(d):
        a, b = SWAP_A[moves[:, t]], SWAP_B[moves[:, t]]
        va, vb = state[rows, a].clone(), state[rows, b].clone()
        state[rows, a], state[rows, b] = vb, va
        states.append(state.clone())
    tokens = torch.cat([moves, torch.full((n, 1), QRY)], 1)
    return tokens, torch.stack(states, 1)


# ---- model ----------------------------------------------------------------
def rope(x, pos):  # x [B,H,T,Dh]
    dh = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, dtype=torch.float32) / dh))
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dm, nh):
        super().__init__()
        self.nh = nh
        self.ln1, self.ln2 = nn.LayerNorm(dm), nn.LayerNorm(dm)
        self.qkv, self.o = nn.Linear(dm, 3 * dm), nn.Linear(dm, dm)
        self.mlp = nn.Sequential(nn.Linear(dm, 4 * dm), nn.GELU(), nn.Linear(4 * dm, dm))
        self.last_attn = None

    def forward(self, x, keep_attn=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.nh, D // self.nh).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T)
        q, k = rope(q, pos), rope(k, pos)
        if keep_attn:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.nh)
            att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
            self.last_attn = att[:, :, -1, :].detach()  # query token's attention [B,H,T]
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class Looped(nn.Module):
    def __init__(self, dm=64, nh=4, n_core=2, norm_anchor=True):
        super().__init__()
        self.emb = nn.Embedding(N_MOVES + 1, dm)
        self.prelude = Block(dm, nh)
        self.adapter = nn.Linear(2 * dm, dm, bias=False)
        with torch.no_grad():  # [0 | I] init, as in the paper
            self.adapter.weight.zero_()
            self.adapter.weight[:, dm:] = torch.eye(dm)
        self.core = nn.ModuleList([Block(dm, nh) for _ in range(n_core)])
        self.ln_f = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, 25)
        self.norm_anchor = norm_anchor

    def forward(self, tokens, R, keep_attn=False):
        """Yields logits [B,5,5] read at the query token after each iteration."""
        e = self.prelude(self.emb(tokens))
        s, ref = e, None
        for k in range(1, R + 1):
            h = self.adapter(torch.cat([e, s], -1))
            for blk in self.core:
                h = blk(h, keep_attn)
            s = h
            if self.norm_anchor:
                rms = s.pow(2).mean(-1, keepdim=True).sqrt()
                if ref is None:
                    ref = rms.detach()
                s = s * (ref / (rms + 1e-6))
            yield self.head(self.ln_f(s[:, -1])).view(-1, 5, 5)


# ---- train / eval -----------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(1000 + args.seed)
    model = Looped(norm_anchor=bool(args.norm))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    t0, log = time.time(), []
    for step in range(args.steps):
        d = int(torch.randint(1, args.train_d + 1, (1,), generator=gen))
        tok, st = make_batch(args.batch, d, gen)
        loss = 0.0
        for k, logits in enumerate(model(tok, d), 1):
            if args.targets == "progress":
                loss = loss + F.cross_entropy(logits.reshape(-1, 5), st[:, k].reshape(-1))
            elif k == d:
                loss = loss + F.cross_entropy(logits.reshape(-1, 5), st[:, d].reshape(-1))
        loss = loss / (d if args.targets == "progress" else 1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            log.append({"step": step, "loss": float(loss), "t": time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
    return model, time.time() - t0, log


@torch.no_grad()
def evaluate(model, depths, n, seed=12345):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for d in depths:
        tok, st = make_batch(n, d, gen)
        per_iter = []
        for k, logits in enumerate(model(tok, d), 1):
            per_iter.append(float((logits.argmax(-1) == st[:, k]).all(-1).float().mean()))
        out[d] = {"full_state_acc": per_iter[-1], "per_iter_acc_vs_zk": per_iter}
        print(d, per_iter[-1], flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", choices=["progress", "final"], default="progress")
    p.add_argument("--norm", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_d", type=int, default=16)
    p.add_argument("--eval_n", type=int, default=512)
    p.add_argument("--depths", default="8,16,32,64,128,256")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.targets}_norm{args.norm}_seed{args.seed}"
    model, wall, log = train(args)
    torch.save(model.state_dict(), os.path.join(args.out, name + ".pt"))
    ev = evaluate(model, [int(x) for x in args.depths.split(",")], args.eval_n)
    json.dump({"args": vars(args), "train_wall_s": wall, "log": log, "eval": ev},
              open(os.path.join(args.out, name + ".json"), "w"))
