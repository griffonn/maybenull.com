"""Figures for the post. Reads results/*.csv, writes figs/*.png (transparent, light+dark safe)."""
import csv, glob, os, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
d = defaultdict(list)
for f in glob.glob("results/grid*.csv"):
    for r in csv.DictReader(open(f)):
        assert r["ok"] == "1", (f, r)
        bits = int(r.get("key_bits") or 64)
        d[(bits, r["dist"], int(r["N"]), int(r["K"]), r["algo"])].append(float(r["ns_per_elem"]))
med = {k: st.median(v) for k, v in d.items()}
lo = {k: min(v) for k, v in d.items()}
hi = {k: max(v) for k, v in d.items()}

INK = "#8a8a85"  # mid-gray: readable on white and on near-black
COL = {"cafs": "#2a78d6", "count_ankerl": "#eb6834", "pdqsort": "#1baf7a", "ska_sort": "#eda100"}
LABEL = {"cafs": "CAFS", "count_ankerl": "hash-map count (ankerl)", "pdqsort": "pdqsort", "ska_sort": "ska_sort (radix)"}
plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.edgecolor": INK, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})
os.makedirs("figs", exist_ok=True)

def Ks(bits, dist, N):
    return sorted({k[3] for k in med if k[:3] == (bits, dist, N)})

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color=INK, alpha=0.15, lw=0.6)
    ax.set_xscale("log")

# Fig 1: absolute ns/element, uniform keys, N=1e7, u64 and i32 side by side.
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
for ax, bits in zip(axes, (64, 32)):
    ks = Ks(bits, "uniform", 10_000_000)
    for a in COL:
        y = [med[(bits, "uniform", 10_000_000, k, a)] for k in ks]
        ax.plot(ks, y, color=COL[a], lw=2, marker="o", ms=4, label=LABEL[a])
    ax.set_yscale("log"); style(ax)
    ax.set_yticks([2, 5, 10, 20, 50]); ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.set_title(f"{'uint64' if bits == 64 else 'int32'} keys, N = 10M, uniform", color=INK)
    ax.set_xlabel("distinct values K")
axes[0].set_ylabel("ns per element (median of 5)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=9)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("figs/ns-per-element.png", dpi=160, transparent=True)

# Fig 2: CAFS time / hash-map-count time, per distribution. >1 means CAFS is slower.
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
DCOL = {"uniform": "#6250d6", "zipf": "#e87ba4", "runs": "#1baf7a"}  # distinct from Fig 1's algorithm colors
for ax, bits in zip(axes, (64, 32)):
    for dist in DCOL:
        ks = Ks(bits, dist, 10_000_000)
        if not ks: continue
        y = [med[(bits, dist, 10_000_000, k, "cafs")] / med[(bits, dist, 10_000_000, k, "count_ankerl")] for k in ks]
        ax.plot(ks, y, color=DCOL[dist], lw=2, marker="o", ms=4, label={"runs": "clustered runs"}.get(dist, dist))
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.set_yscale("log"); style(ax)
    ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
    ax.set_title(f"{'uint64' if bits == 64 else 'int32'} keys, N = 10M", color=INK)
    ax.set_xlabel("distinct values K")
axes[0].set_ylabel("CAFS time ÷ hash-map-count time\n(below 1 = CAFS faster)")
axes[0].legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("figs/cafs-vs-count.png", dpi=160, transparent=True)
print("wrote figs/")
