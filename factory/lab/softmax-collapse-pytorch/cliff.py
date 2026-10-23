"""Where does PyTorch's cross-entropy zero the target gradient? Sweep the logit margin.

K classes; target logit = m + log(K-1), others 0, so the true margin is exactly m.
For each dtype/path we record the target gradient, the gradient sum, and the loss.
Writes results/cliff.csv.
"""
import csv, math, sys, platform
import torch, torch.nn.functional as F
from ce import fixed_ce, fixed_ce_fn

K = 32
t = torch.tensor([0])
rows = []
ms = [i / 20 for i in range(0, 20 * 100 + 1)]  # 0 .. 100 step 0.05


def run(path, dtype, m):
    z0 = torch.zeros(1, K, dtype=torch.float64)
    z0[0, 0] = m + math.log(K - 1)
    if path == "autocast_bf16":
        # fp32 leaf -> bf16 matmul output under autocast -> F.cross_entropy (autocast runs it in fp32)
        w = torch.eye(K, dtype=torch.float32, requires_grad=True)
        x = z0.float()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            z = x @ w
            loss = F.cross_entropy(z, t)
        z.retain_grad()
        loss.backward()
        g = z.grad.double()[0]
        return loss.item(), g, str(z.dtype), str(loss.dtype)
    z = z0.to(dtype).requires_grad_(True)
    if path == "F.cross_entropy":
        loss = F.cross_entropy(z, t)
    elif path == "log_softmax+nll":
        loss = F.nll_loss(F.log_softmax(z, -1), t)
    elif path == "fixed":
        loss = fixed_ce(z, t)
    elif path == "fixed_fn":
        loss = fixed_ce_fn(z, t)
    loss.backward()
    return loss.item(), z.grad.double()[0], str(z.dtype), str(loss.dtype)


cases = [(p, d) for p in ["F.cross_entropy", "log_softmax+nll", "fixed", "fixed_fn"]
         for d in [torch.float32, torch.bfloat16, torch.float16, torch.float64]]
cases.append(("autocast_bf16", None))
for p, d in cases:
    for m in ms:
        loss, g, zd, ld = run(p, d, m)
        true_gt = -math.exp(-m) / (1 + math.exp(-m))
        rows.append(dict(path=p, dtype=zd, loss_dtype=ld, m=m, loss=loss, g_t=g[0].item(),
                         g_other_sum=g[1:].sum().item(), g_sum=g.sum().item(), true_g_t=true_gt))
with open("results/cliff.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
# summary: first margin at which g_t == 0 while others are nonzero
print(f"torch {torch.__version__} python {platform.python_version()}")
for p, d in cases:
    sub = [r for r in rows if r["path"] == p and (d is None or r["dtype"] == str(d))]
    cl = next((r["m"] for r in sub if r["g_t"] == 0.0 and r["g_other_sum"] != 0.0), None)
    print(f"{p:18s} {sub[0]['dtype']:15s} loss:{sub[0]['loss_dtype']:15s} target-grad cliff at m = {cl}")
