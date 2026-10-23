"""Summarize results/grok/*.json and results/digits/*.json -> results/summary.json + markdown tables.

Spike definition (fixed before the full grid ran): a logged step whose exact fp64 train CE is
more than 10x the minimum over the preceding 500 steps; consecutive such steps count as one spike.
Test-accuracy dip: largest drop of test accuracy below its running maximum, after the run first
reached 90% test accuracy.
"""
import glob, json, os, statistics as st
from collections import defaultdict

ARMS = ["std_fp32", "fixed_fp32", "std_fp32_eps1e-7", "std_fp32_ls0.01", "std_fp32_loss64",
        "std_autocast_bf16", "std_bf16logits", "fixed_bf16logits", "std_bf16", "fixed_bf16"]


def spikes(log, window=50):
    n, inside = 0, False
    for i in range(1, len(log)):
        lo = min(r[1] for r in log[max(0, i - window):i])
        hit = log[i][1] > 10 * lo
        if hit and not inside:
            n += 1
        inside = hit
    return n


def first_step(log, col, thr):
    return next((r[0] for r in log if r[col] >= thr), None)


def dip(log):
    best, worst, started = 0.0, 0.0, False
    for r in log:
        if r[7] >= 0.9:
            started = True
        best = max(best, r[7])
        if started:
            worst = max(worst, best - r[7])
    return worst


def ms(x):
    x = [v for v in x if v is not None]
    if not x:
        return None
    return dict(mean=st.mean(x), sd=st.stdev(x) if len(x) > 1 else 0.0, min=min(x), max=max(x), n=len(x))


runs = defaultdict(list)
for f in sorted(glob.glob("results/grok/*.json")):
    d = json.load(open(f)); runs[d["arm"]].append(d)
summ = {"grok": {}, "digits": {}}
for a in ARMS:
    if a not in runs:
        continue
    R = runs[a]
    per = [dict(seed=d["seed"], final_test=d["log"][-1][7], final_margin=d["log"][-1][3],
                max_dead=max(r[5] for r in d["log"]), spikes=spikes(d["log"]),
                t99=first_step(d["log"], 7, 0.99), t_mem=first_step(d["log"], 6, 1.0), dip=dip(d["log"]),
                seconds=d["seconds"]) for d in sorted(R, key=lambda d: d["seed"])]
    summ["grok"][a] = dict(per_seed=per, final_test=ms([p["final_test"] for p in per]),
                           final_margin=ms([p["final_margin"] for p in per]), spikes=ms([p["spikes"] for p in per]),
                           t99=ms([p["t99"] for p in per]), reached99=sum(p["t99"] is not None for p in per),
                           dip=ms([p["dip"] for p in per]), max_dead=ms([p["max_dead"] for p in per]))
dg = defaultdict(list)
for f in sorted(glob.glob("results/digits/*.json")):
    d = json.load(open(f)); dg[(d["loss"], d["wd"])].append(d)
for (loss, wd), R in sorted(dg.items()):
    summ["digits"][f"{loss}_wd{wd}"] = dict(
        final_test=ms([d["log"][-1][5] for d in R]), final_median_margin=ms([d["log"][-1][2] for d in R]),
        final_min_margin=ms([d["log"][-1][3] for d in R]), max_dead=ms([max(r[4] for r in d["log"]) for d in R]),
        first_dead_step=ms([next((r[0] for r in d["log"] if r[4] > 0), None) for d in R]),
        spikes=ms([spikes([[r[0], r[1]] for r in d["log"]], window=10) for d in R]), n=len(R))
json.dump(summ, open("results/summary.json", "w"), indent=1)

print("| arm | final test acc % (mean ± sd, min) | seeds reaching 99% | steps to 99% (mean) | loss spikes / run | worst test-acc dip, pts | final mean margin |")
print("|---|---|---|---|---|---|---|")
for a, s in summ["grok"].items():
    ft, t9, sp, dp, fm = s["final_test"], s["t99"], s["spikes"], s["dip"], s["final_margin"]
    print(f"| {a} | {100*ft['mean']:.1f} ± {100*ft['sd']:.1f} (min {100*ft['min']:.1f}) | {s['reached99']}/{ft['n']} | "
          f"{'%.0f' % t9['mean'] if t9 else '—'} | {sp['mean']:.1f} (max {sp['max']}) | {100*dp['max']:.1f} | {fm['mean']:.1f} |")
for k, s in summ["digits"].items():
    print(k, {kk: (round(v["mean"], 4), round(v["min"], 4), round(v["max"], 4)) if isinstance(v, dict) else v for kk, v in s.items()})
