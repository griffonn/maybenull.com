"""Figures: MAE vs N per arm (balanced protocol), and pred-vs-true scatter at N=100, seed 0."""
import collections, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#888", "axes.labelcolor": "#666",
                     "xtick.color": "#666", "ytick.color": "#666", "text.color": "#666",
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"pair": "#d1495b", "mse": "#2e86ab", "l1": "#66a182", "mse_raw": "#7b5ea7", "ridge_pixels": "#edae49", "const_train_mean": "#999999"}
LBL = {"pair": "pairwise CE + entropy decode (paper)", "mse": "pointwise MSE, standardized target",
       "l1": "pointwise L1, standardized target", "mse_raw": "pointwise MSE, raw age (no standardization)", "ridge_pixels": "ridge regression on raw pixels",
       "const_train_mean": "constant: training-set mean"}

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

fig, ax = plt.subplots(figsize=(7.2, 4.4))
for arm in ["const_train_mean", "ridge_pixels", "pair", "mse", "l1", "mse_raw"]:
    ns = sorted(n for (a, s, n) in runs if a == arm and s == "balanced")
    if not ns:
        continue
    m = [np.mean([d["final_test_mae"] for d in runs[(arm, "balanced", n)]]) for n in ns]
    lo = [np.min([d["final_test_mae"] for d in runs[(arm, "balanced", n)]]) for n in ns]
    hi = [np.max([d["final_test_mae"] for d in runs[(arm, "balanced", n)]]) for n in ns]
    ls = "--" if arm in ("const_train_mean", "ridge_pixels") else "-"
    ax.plot(ns, m, ls, marker="o", color=C[arm], label=LBL[arm], lw=2)
    ax.fill_between(ns, lo, hi, color=C[arm], alpha=0.15, lw=0)
ax.axhline(22.6, color="#999", lw=1, ls=":")
ax.text(105, 22.9, "paper's reported MSE baseline at N=100 (22.6)", fontsize=9, color="#888")
ax.set_xscale("log"); ax.set_xticks([100, 300, 1000]); ax.set_xticklabels(["100", "300", "1000"])
ax.set_xlabel("labeled training images (decade-balanced, as in the paper's code)")
ax.set_ylabel("test MAE, years (lower is better)")
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.93))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mae-vs-n.png"), dpi=160, transparent=True)

fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)
for ax, arm in zip(axs, ["mse", "pair"]):
    d = json.load(open(os.path.join(R, "long", f"{arm}_balanced_n100_s0.json")))  # 10k-step runs
    t, p = np.array(d["true"]), np.array(d["pred"])
    ax.scatter(t, p, s=12, color=C[arm], alpha=0.7)
    ax.plot([0, 100], [0, 100], color="#999", lw=1)
    ax.axhline(d["train_mean"], color="#999", lw=1, ls=":")
    ax.set_title(f"{'pointwise MSE' if arm == 'mse' else 'pairwise'}: MAE {np.abs(t - p).mean():.1f}", fontsize=11)
    ax.set_xlabel("true age")
axs[0].set_ylabel("predicted age (N=100, 10k steps, seed 0)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "scatter-n100.png"), dpi=160, transparent=True)
print("wrote figs")
