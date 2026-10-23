"""How does the fp32/bf16 cliff of F.cross_entropy depend on the number of classes K
and on how the non-target mass is spread? Writes results/cliff_k.csv.

'spread': all K-1 non-target logits equal.  'one': one runner-up holds all non-target mass,
the rest sit 60 below it (negligible).  True margin is exactly m in both cases.
"""
import csv, math
import torch, torch.nn.functional as F

rows = []
for dtype in [torch.float32, torch.bfloat16, torch.float16, torch.float64]:
    for K in [2, 8, 32, 1024, 32768, 131072]:
        for shape in ["spread", "one"]:
            if shape == "one" and K == 2:
                continue
            cliff = None
            hi = 80 if dtype == torch.float64 else 30
            for i in range(0, hi * 100 + 1):
                m = i / 100
                z = torch.zeros(1, K, dtype=torch.float64)
                if shape == "spread":
                    z[0, 0] = m + math.log(K - 1)
                else:
                    z[0, 2:] = -60.0
                    z[0, 0] = m
                z = z.to(dtype).requires_grad_(True)
                F.cross_entropy(z, torch.tensor([0])).backward()
                g = z.grad
                if g[0, 0].item() == 0.0 and g[0, 1:].sum().item() != 0.0:
                    cliff = m
                    break
            rows.append(dict(dtype=str(dtype).replace("torch.", ""), K=K, shape=shape, cliff=cliff))
            print(rows[-1], flush=True)
with open("results/cliff_k.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
