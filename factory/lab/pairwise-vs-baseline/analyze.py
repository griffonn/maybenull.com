"""Summarize results/*.json + results/baselines.json into a markdown table and results/summary.csv."""
import collections, csv, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
runs = collections.defaultdict(list)
for f in glob.glob(os.path.join(R, "*.json")) + glob.glob(os.path.join(R, "long", "*.json")):
    if f.endswith("baselines.json"):
        continue
    d = json.load(open(f)); a = d["args"]
    pred, true = np.array(d["pred"]), np.array(d["true"])
    arm = a["arm"] + ("" if a["steps"] == 3000 else f" ({a['steps'] // 1000}k steps)")
    runs[(arm, a["sampling"], a["n"])].append({
        "final": d["final_test_mae"], "best_on_test": d["best_on_test_mae"],
        "pred_sd": pred.std(), "true_sd": true.std(), "wall": d["wall_s"]})
base = json.load(open(os.path.join(R, "baselines.json")))
for r in base["rows"]:
    for k in ["const_train_mean", "ridge_pixels"]:
        runs[(k, r["sampling"], r["n"])].append({"final": r[k], "best_on_test": np.nan, "pred_sd": np.nan,
                                                  "true_sd": np.nan, "wall": 0})
rows = []
print("| arm | train sampling | N | seeds | test MAE (final) | MAE if checkpoint picked on test | pred SD / true SD |")
print("|---|---|---|---|---|---|---|")
for key in sorted(runs, key=lambda k: (k[1], k[2], k[0])):
    v = runs[key]
    f = np.array([x["final"] for x in v]); b = np.array([x["best_on_test"] for x in v])
    sdr = np.array([x["pred_sd"] / x["true_sd"] for x in v])
    row = dict(arm=key[0], sampling=key[1], n=key[2], seeds=len(v), mae_mean=f.mean(), mae_sd=f.std(ddof=1) if len(v) > 1 else 0,
               mae_min=f.min(), mae_max=f.max(), best_on_test_mean=np.nanmean(b) if not np.isnan(b).all() else np.nan,
               sd_ratio=np.nanmean(sdr) if not np.isnan(sdr).all() else np.nan, wall_mean=np.mean([x["wall"] for x in v]))
    rows.append(row)
    print(f"| {key[0]} | {key[1]} | {key[2]} | {len(v)} | {row['mae_mean']:.2f} ± {row['mae_sd']:.2f} "
          f"({row['mae_min']:.1f}–{row['mae_max']:.1f}) | {row['best_on_test_mean']:.2f} | {row['sd_ratio']:.2f} |")
with open(os.path.join(R, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("constant (whole-dataset median) MAE on the test set:", round(base["const_dataset_median"], 2))
