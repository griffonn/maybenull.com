"""Summarise agent runs and simulations into results/summary.json and print tables.

usage: python analyze.py
"""
import csv, glob, json
import numpy as np

def stop_round(log, p):
    run = 0
    for e in log[1:]:
        run = 0 if e["accepted"] else run + 1
        if run >= p:
            return e["t"]
    return log[-1]["t"]

def at(log, t):
    e = log[t - 1]
    return 1 - e["champ_val"], 1 - e["champ_test"]

summary = {"agent": {}, "sim_local": {}, "sim_boost": {}}
print("searcher task   rule   | accepts | val gain  test gain  inflation (val gain - test gain) | stop@p=5: round inflation")
for searcher in ["tweak", "ensemble"]:
    for task in ["null", "signal"]:
        for rule in ["exact", "pse", "ladder"]:
            runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/agent_{searcher}_{task}_{rule}_*.json"))]
            if not runs:
                continue
            row = {}
            fin = np.array([at(r["log"], len(r["log"])) for r in runs])
            start = np.array([at(r["log"], 1) for r in runs])
            row["n_seeds"] = len(runs)
            row["accepts"] = [sum(e["accepted"] for e in r["log"][1:]) for r in runs]
            row["final_val"], row["final_test"] = fin[:, 0].tolist(), fin[:, 1].tolist()
            row["start_val"], row["start_test"] = start[:, 0].tolist(), start[:, 1].tolist()
            row["median_m"] = float(np.median([e["m"] for r in runs for e in r["log"][1:]]))
            for p in [3, 5, 10, 20]:
                st = [stop_round(r["log"], p) for r in runs]
                vt = np.array([at(r["log"], s) for r, s in zip(runs, st)])
                row[f"p{p}"] = dict(stop=st, val=vt[:, 0].tolist(), test=vt[:, 1].tolist())
            # curves for plotting: mean champion val/test acc after each submission
            row["curve_val"] = np.mean([[1 - e["champ_val"] for e in r["log"]] for r in runs], 0).tolist()
            row["curve_test"] = np.mean([[1 - e["champ_test"] for e in r["log"]] for r in runs], 0).tolist()
            summary["agent"][f"{searcher}/{task}/{rule}"] = row
            # gains are measured from the first (start) model; inflation = how much of the
            # validation gain does not show up on the fresh test set
            vg, tg = fin[:, 0] - start[:, 0], fin[:, 1] - start[:, 1]
            row["val_gain"], row["test_gain"], row["inflation"] = vg.tolist(), tg.tolist(), (vg - tg).tolist()
            for p in [3, 5, 10, 20]:
                q = row[f"p{p}"]
                q["inflation"] = ((np.array(q["val"]) - start[:, 0]) - (np.array(q["test"]) - start[:, 1])).tolist()
            p5 = row["p5"]
            pct = lambda x: f"{np.mean(x)*100:+.2f}±{np.std(x, ddof=1)*100:.2f}"
            print(f"{searcher:8s} {task:6s} {rule:6s} | {np.mean(row['accepts']):5.1f}   | "
                  f"{pct(vg)}  {pct(tg)}  {pct(vg - tg)} | {np.mean(p5['stop']):5.1f} {pct(p5['inflation'])}")

try:
    rows = list(csv.DictReader(open("results/sim_local.csv")))
    print("\nsim local (n=25000, 1000 submissions, pure noise): val gain / test gain / accepts")
    for rule in ["exact", "pse", "pse1", "ladder"]:
        for m in sorted({r["m"] for r in rows}, key=int):
            rr = [r for r in rows if r["rule"] == rule and r["m"] == m]
            vg = np.array([float(r["val_gain_final"]) for r in rr]); tg = np.array([float(r["test_gain_final"]) for r in rr])
            ac = np.array([int(r["accepts"]) for r in rr])
            summary["sim_local"][f"{rule}/{m}"] = dict(val_gain=vg.tolist(), test_gain=tg.tolist(), accepts=ac.tolist(),
                curve=np.mean([[float(x) for x in r["curve"].split()] for r in rr], 0)[::10].tolist())
            print(f"  {rule:6s} m={m:>5s}  {vg.mean():.4f}±{vg.std(ddof=1):.4f}  {tg.mean():+.4f}  {ac.mean():6.1f}")
    summary["sim_firststep"] = {r["m"]: float(r["pass_rate"]) for r in csv.DictReader(open("results/sim_firststep.csv"))}
    print("\nfirst-step pass rate of pse (unselected champion):", summary["sim_firststep"])
    rows = list(csv.DictReader(open("results/sim_boost.csv")))
    print("\nsim boost (Blum-Hardt attack, 1000 random vectors): val gain / test gain / vectors used")
    for rule in ["exact", "pse", "ladder"]:
        rr = [r for r in rows if r["rule"] == rule]
        vg = np.array([float(r["val_gain"]) for r in rr]); tg = np.array([float(r["test_gain"]) for r in rr])
        summary["sim_boost"][rule] = dict(val_gain=vg.tolist(), test_gain=tg.tolist(), used=[int(r["n_used"]) for r in rr])
        print(f"  {rule:6s} {vg.mean():.4f}±{vg.std(ddof=1):.4f}  {tg.mean():+.4f}  {np.mean([int(r['n_used']) for r in rr]):.0f}")
except FileNotFoundError:
    pass

n = 25_000
summary["floor"] = {k: float(n ** (-1 / 3) * np.log(k * n) ** (1 / 3)) for k in [30, 200, 1000]}
print("\nfloor eta_min (c=1, n=25000):", {k: round(v, 4) for k, v in summary["floor"].items()})
json.dump(summary, open("results/summary.json", "w"))
