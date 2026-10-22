#!/usr/bin/env python3
"""Aggregate results/*.jsonl into results/summary.json and print markdown tables. Stdlib only."""
import json, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ORDER = ["uniform", "zipf_spread", "zipf_contig", "hot_spread", "stride81", "stride27", "stride9",
         "stride3", "hot_contig", "walk", "sequential", "sequential_desc"]
METRICS = ["poptree", "poptree_median", "poptree_level", "splay", "static", "entropy_static", "opt", "opt_term1", "opt_term2",
           "restructure_ops", "min_alpha", "min_alpha_median"]


def load(name):
    with open(os.path.join(RES, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def agg(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["workload"]].append(r)
    out = {}
    for w, rs in by.items():
        d = {"seeds": len(rs), "n": rs[0]["n"], "m": rs[0]["m"]}
        for k in METRICS:
            v = [r[k] for r in rs]
            d[k] = {"mean": st.mean(v), "min": min(v), "max": max(v)}
        out[w] = d
    return out


def main():
    summary = {}
    check = load("check_L8.jsonl")
    summary["check_L8"] = {"all_checked": all(r["checked"] for r in check),
                           "min_alpha": min(r["min_alpha"] for r in check),
                           "min_alpha_median": min(r["min_alpha_median"] for r in check),
                           "workloads": len(check), "m": check[0]["m"], "n": check[0]["n"]}
    for name in ["grid_L10", "grid_L12"]:
        p = os.path.join(RES, name + ".jsonl")
        if os.path.exists(p):
            summary[name] = agg(load(name + ".jsonl"))
    # rank correlation between OPT(X) (bits, constants dropped) and PopTree's nodes touched
    for name in ["grid_L10", "grid_L12"]:
        if name in summary:
            g = summary[name]; ws = sorted(g)
            b = [g[w]["opt"]["mean"] for w in ws]
            rank = lambda v: [sorted(v).index(x) for x in v]
            k = len(ws)
            for metric in ["poptree", "poptree_level"]:
                a = [g[w][metric]["mean"] for w in ws]
                summary[f"{name}_opt_spearman_{metric}"] = 1 - 6 * sum((x - y) ** 2 for x, y in zip(rank(a), rank(b))) / (k * (k * k - 1))
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("invariant check:", summary["check_L8"])
    for name in ["grid_L10", "grid_L12"]:
        for metric in ["poptree", "poptree_level"]:
            key = f"{name}_opt_spearman_{metric}"
            if key in summary:
                print(f"{name}: Spearman(OPT, {metric}) over workloads = {summary[key]:.3f}")
    for name in ["grid_L10", "grid_L12"]:
        if name not in summary:
            continue
        g = summary[name]
        any_w = next(iter(g.values()))
        print(f"\n### {name}: n={any_w['n']}, m={any_w['m']}, seeds={any_w['seeds']} "
              "(nodes touched per search, mean [min-max over seeds]; OPT in bits, constants dropped)")
        print("| workload | PopTree (paper layout) | PopTree (median layout) | skip-list level | splay | static balanced | offline entropy BST | PopTree/static | OPT(X) term | restructure ops |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for w in ORDER:
            if w not in g:
                continue
            d = g[w]
            f = lambda k: f"{d[k]['mean']:.2f} [{d[k]['min']:.2f}-{d[k]['max']:.2f}]"
            print(f"| {w} | {f('poptree')} | {f('poptree_median')} | {f('poptree_level')} | {f('splay')} | {f('static')} | "
                  f"{f('entropy_static')} | {d['poptree']['mean'] / d['static']['mean']:.2f} | {d['opt']['mean']:.2f} | "
                  f"{d['restructure_ops']['mean']:.1f} |")


if __name__ == "__main__":
    main()
