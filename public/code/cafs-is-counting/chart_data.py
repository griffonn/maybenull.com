"""Chart data for the post's interactive charts. Reads results/*.csv like plot.py,
writes figs/ns-per-element.json and figs/cafs-vs-count.json."""
import csv, glob, json, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
d = defaultdict(list)
for f in glob.glob("results/grid*.csv"):
    for r in csv.DictReader(open(f)):
        assert r["ok"] == "1", (f, r)
        bits = int(r.get("key_bits") or 64)
        d[(bits, r["dist"], int(r["N"]), int(r["K"]), r["algo"])].append(float(r["ns_per_elem"]))
med = {k: st.median(v) for k, v in d.items()}
N = 10_000_000

def Ks(bits, dist):
    return sorted({k[3] for k in med if k[:3] == (bits, dist, N)})

os.makedirs("figs", exist_ok=True)
fig1 = {"panels": []}
for bits in (64, 32):
    ks = Ks(bits, "uniform")
    fig1["panels"].append({"title": f"{'uint64' if bits == 64 else 'int32'} keys, N = 10M, uniform", "K": ks,
        "series": {a: {"median": [round(med[(bits, "uniform", N, k, a)], 3) for k in ks],
                       "min": [round(min(d[(bits, "uniform", N, k, a)]), 3) for k in ks],
                       "max": [round(max(d[(bits, "uniform", N, k, a)]), 3) for k in ks]}
                   for a in ("cafs", "count_ankerl", "pdqsort", "ska_sort")}})
json.dump(fig1, open("figs/ns-per-element.json", "w"))

fig2 = {"panels": []}
for bits in (64, 32):
    p = {"title": f"{'uint64' if bits == 64 else 'int32'} keys, N = 10M", "series": {}}
    for dist in ("uniform", "zipf", "runs"):
        ks = Ks(bits, dist)
        if ks:
            p["series"][dist] = {"K": ks, "ratio": [round(med[(bits, dist, N, k, "cafs")] / med[(bits, dist, N, k, "count_ankerl")], 4) for k in ks]}
    fig2["panels"].append(p)
json.dump(fig2, open("figs/cafs-vs-count.json", "w"))
print("wrote figs/*.json")
