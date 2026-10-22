"""Figures for the post. Transparent PNGs with mid-gray ink, readable on light and dark pages.

usage: python plot.py   (needs results/summary.json from analyze.py)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("results/summary.json"))
INK, GRID = "#8a8a86", "#8a8a8640"
C = {"exact": "#eb6834", "pse": "#2a78d6", "ladder": "#1baf7a"}
LBL = {"exact": "exact score", "pse": "1 paired SE (paper)", "ladder": "fixed margin 0.005"}
plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: fake gain after 1000 pure-noise submissions vs edit size m
fig, ax = plt.subplots(figsize=(7.5, 5.0))
for rule in ["exact", "pse", "ladder"]:
    ms = sorted({int(k.split("/")[1]) for k in S["sim_local"] if k.startswith(rule + "/")})
    v = np.array([np.mean(S["sim_local"][f"{rule}/{m}"]["val_gain"]) * 100 for m in ms])
    e = np.array([np.std(S["sim_local"][f"{rule}/{m}"]["val_gain"], ddof=1) * 100 for m in ms])
    ax.plot(ms, v, "-o", color=C[rule], lw=2, ms=5, label=LBL[rule])
    ax.fill_between(ms, v - e, v + e, color=C[rule], alpha=0.15, lw=0)
ms = sorted({int(k.split("/")[1]) for k in S["sim_local"] if k.startswith("pse1/")})
v1 = [np.mean(S["sim_local"][f"pse1/{m}"]["val_gain"]) * 100 for m in ms]
ax.plot(ms, v1, "--", color=C["pse"], lw=1.2, label="1 paired SE, sample std")
ms = sorted({int(k.split("/")[1]) for k in S["sim_local"] if k.startswith("exact/")})
t = [np.mean(S["sim_local"][f"exact/{m}"]["test_gain"]) * 100 for m in ms]
ax.plot(ms, t, ":", color=INK, lw=1.5, label="fresh-test gain (exact; other rules similar)")
ax.set_xscale("log")
ax.set_xlabel("predictions changed per submission (m), validation n = 25,000")
ax.set_ylabel("reported accuracy gain over 50%\n(percentage points)")
ax.set_title("Pure noise, 1,000 submissions: what each harness lets through", color=INK, fontsize=12)
ax.grid(True, color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.tight_layout(); fig.savefig("figs/noise-sweep.png", dpi=150, transparent=True)

print("ok")

# Chart data for the post's interactive Plotly chart: figs/noise-sweep.json
out = {"m": sorted({int(k.split("/")[1]) for k in S["sim_local"] if k.startswith("exact/")}), "series": {}}
for rule in ["exact", "pse", "pse1", "ladder"]:
    out["series"][rule] = {
        "mean": [round(float(np.mean(S["sim_local"][f"{rule}/{m}"]["val_gain"])) * 100, 3) for m in out["m"]],
        "sd": [round(float(np.std(S["sim_local"][f"{rule}/{m}"]["val_gain"], ddof=1)) * 100, 3) for m in out["m"]]}
out["series"]["test_exact"] = {"mean": [round(float(np.mean(S["sim_local"][f"exact/{m}"]["test_gain"])) * 100, 3) for m in out["m"]]}
json.dump(out, open("figs/noise-sweep.json", "w"), indent=1)
print("json ok")
