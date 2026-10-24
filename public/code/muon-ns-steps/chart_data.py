"""Write the numbers the post's Plotly charts plot to figs/*.json
(copied to public/lab/muon-ns-steps/ on the site)."""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)

f = json.load(open(os.path.join(HERE, "results/filter_curves.json")))
keep = list(range(len(f["s"])))  # all 1200 points: the NS curves oscillate, so no downsampling
pick = lambda v: [round(v[i], 5) for i in keep]
json.dump({"s": [f["s"][i] for i in keep],
           "ns": {q: pick(f["ns"][q]) for q in ["1", "2", "3", "5", "8"]},
           "damped": {q: pick(f["damped"][q]) for q in ["1", "3", "5"]},
           "inv_l5": 3.4445 ** -5}, open(os.path.join(OUT, "filters.json"), "w"))

s = json.load(open(os.path.join(HERE, "results/summary.json")))
json.dump({arm: {"lr": v["lr"], "vals": [round(x, 4) for x in v["vals"]], "seeds": v["seeds"],
                 "mean": round(v["mean"], 4)} for arm, v in s["seeds"].items()},
          open(os.path.join(OUT, "final-loss.json"), "w"))

long = {}
for fn in sorted(glob.glob(os.path.join(HERE, "results/long/*.json"))):
    r = json.load(open(fn))
    long.setdefault(r["arm"], {})[r["seed"]] = {"step": [e["step"] for e in r["log"]], "val": [round(e["val"], 4) for e in r["log"]]}
json.dump(long, open(os.path.join(OUT, "long-curves.json"), "w"))
print("wrote", os.listdir(OUT))
