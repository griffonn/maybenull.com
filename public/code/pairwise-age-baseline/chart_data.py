"""Chart data for the post's interactive charts. Reads results/ like plot.py,
writes figs/mae-vs-n.json and figs/scatter-n100.json."""
import collections, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
runs = collections.defaultdict(list)
for f in glob.glob(os.path.join(R, "*.json")):
    if f.endswith(("baselines.json", "const_random_test.json")):
        continue
    d = json.load(open(f)); a = d["args"]
    runs[(a["arm"], a["sampling"], a["n"])].append(d)
base = json.load(open(os.path.join(R, "baselines.json")))
for r in base["rows"]:
    for k in ["const_train_mean", "ridge_pixels"]:
        runs[(k, r["sampling"], r["n"])].append({"final_test_mae": r[k]})

fig1 = {"paper_mse_baseline": 22.6, "arms": {}}
for arm in ["const_train_mean", "ridge_pixels", "pair", "mse", "l1", "mse_raw"]:
    ns = sorted(n for (a, s, n) in runs if a == arm and s == "balanced")
    if not ns:
        continue
    v = [[d["final_test_mae"] for d in runs[(arm, "balanced", n)]] for n in ns]
    fig1["arms"][arm] = {"n": ns, "mean": [round(float(np.mean(x)), 3) for x in v],
                         "min": [round(float(np.min(x)), 3) for x in v], "max": [round(float(np.max(x)), 3) for x in v],
                         "seeds": [len(x) for x in v]}
json.dump(fig1, open(os.path.join(OUT, "mae-vs-n.json"), "w"))

fig2 = {}
for arm in ["mse", "pair"]:
    d = json.load(open(os.path.join(R, "long", f"{arm}_balanced_n100_s0.json")))  # 10k-step runs
    t, p = np.array(d["true"]), np.array(d["pred"])
    fig2[arm] = {"true": t.round(2).tolist(), "pred": p.round(2).tolist(), "train_mean": round(float(d["train_mean"]), 2),
                 "mae": round(float(np.abs(t - p).mean()), 2)}
json.dump(fig2, open(os.path.join(OUT, "scatter-n100.json"), "w"))
print("wrote figs/*.json")
