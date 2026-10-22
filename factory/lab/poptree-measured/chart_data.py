#!/usr/bin/env python3
"""Chart data for the post's interactive charts. Reads results/summary.json (run analyze.py
first), writes figs/by-workload.json and figs/stride.json. Stdlib only."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(HERE, "results", "summary.json")))
ARMS = ["poptree", "splay", "static", "entropy_static"]


def pack(g, workloads):
    out = {}
    for a in ARMS:
        out[a] = {k: [round(g[w][a][k], 3) for w in workloads] for k in ["mean", "min", "max"]}
    return out


g = S["grid_L10"]
WL = [("uniform", "uniform"), ("zipf_spread", "Zipf, keys shuffled"), ("zipf_contig", "Zipf, keys adjacent"),
      ("hot_spread", "729 hot keys, random spots"), ("hot_contig", "729 hot keys, adjacent"),
      ("walk", "random walk ±8"), ("sequential", "scan, ascending"), ("sequential_desc", "scan, descending")]
fig1 = {"n": next(iter(g.values()))["n"], "m": next(iter(g.values()))["m"],
        "seeds": next(iter(g.values()))["seeds"], "workloads": [w for w, _ in WL],
        "labels": [l for _, l in WL], "arms": pack(g, [w for w, _ in WL])}
json.dump(fig1, open(os.path.join(OUT, "by-workload.json"), "w"))

ST = [(1, "hot_contig"), (3, "stride3"), (9, "stride9"), (27, "stride27"), (81, "stride81")]
fig2 = {"n": fig1["n"], "hot_keys": 729, "seeds": fig1["seeds"], "spacing": [s for s, _ in ST],
        "arms": pack(g, [w for _, w in ST]),
        "opt_bits": [round(g[w]["opt"]["mean"], 3) for _, w in ST]}
json.dump(fig2, open(os.path.join(OUT, "stride.json"), "w"))
print("wrote", os.listdir(OUT))
