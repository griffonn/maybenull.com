"""Summarize sweep + seeds into results/summary.json and print tables."""
import collections, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["adamw", "ns1", "ns2", "ns3", "ns5", "ns8", "polar", "damped1", "damped3", "damped5"]

def load(sub):
    out = collections.defaultdict(list)
    for f in glob.glob(os.path.join(HERE, "results", sub, "*.json")):
        r = json.load(open(f)); out[(r["arm"], r["lr"], r["steps"])].append(r)
    return out

sweep = load("sweep")
summary = {"sweep": {}, "best_lr": {}, "seeds": {}}
print("LR sweep (seed 0, 600 steps): final val loss")
for arm in ARMS:
    rows = sorted((lr, rs[0]["final_val"]) for (a, lr, st), rs in sweep.items() if a == arm and st == 600)
    if not rows: continue
    summary["sweep"][arm] = rows
    best = min(rows, key=lambda x: x[1] if np.isfinite(x[1]) else 9e9)
    summary["best_lr"][arm] = best[0]
    print(f"  {arm:8s} " + "  ".join(f"{lr:g}:{v:.4f}" for lr, v in rows) + f"   best {best[0]:g}")
seeds = load("sweep"); seeds2 = load("seeds")
for k, v in seeds2.items(): seeds[k] += v
print("\nAt best LR, all seeds:")
for arm, lr in summary["best_lr"].items():
    rs = seeds.get((arm, lr, 600), [])
    v = np.array([r["final_val"] for r in rs]); 
    opt = np.median([r["opt_time"] / r["steps"] * 1000 for r in rs])
    summary["seeds"][arm] = dict(lr=lr, vals=v.tolist(), seeds=[r["seed"] for r in rs], mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                                 opt_ms_per_step_median=float(opt), curves={r["seed"]: [(e["step"], e["val"]) for e in r["log"]] for r in rs})
    print(f"  {arm:8s} lr={lr:g} n={len(v)} mean={v.mean():.4f} sd={(v.std(ddof=1) if len(v)>1 else float('nan')):.4f} vals={np.round(v,4).tolist()} opt={opt:.1f} ms/step")
json.dump(summary, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)
