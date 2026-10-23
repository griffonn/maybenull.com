"""Forward+backward time of fused F.cross_entropy vs the fixed CE (autograd version and
custom autograd.Function), fp32 and bf16, CPU, 4 threads. Rows = tokens. Writes results/speed.csv.
"""
import csv, time, platform
import torch, torch.nn.functional as F
from ce import fixed_ce, fixed_ce_fn

torch.set_num_threads(4)
torch.manual_seed(0)
fns = {"F.cross_entropy": F.cross_entropy, "fixed (autograd)": fixed_ce, "fixed (autograd.Function)": fixed_ce_fn}
rows = []
for dtype in [torch.float32, torch.bfloat16]:
    for V in [1000, 32000, 128000]:
        N = 1024 if V < 100000 else 512
        z0 = (torch.randn(N, V) * 3).to(dtype)
        t = torch.randint(0, V, (N,))
        for name, f in fns.items():
            z = z0.clone().requires_grad_(True)
            for _ in range(2):
                f(z, t).backward(); z.grad = None
            ts = []
            for _ in range(7):
                t0 = time.perf_counter(); f(z, t).backward(); ts.append(time.perf_counter() - t0); z.grad = None
            ts.sort()
            rows.append(dict(dtype=str(dtype).replace("torch.", ""), vocab=V, rows=N, impl=name,
                             median_ms=round(ts[3] * 1000, 2), min_ms=round(ts[0] * 1000, 2)))
            print(rows[-1], flush=True)
with open("results/speed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
