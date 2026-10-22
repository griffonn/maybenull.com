"""Model-free baselines on the same splits: constant predictors and ridge on raw pixels."""
import json, os, sys
import numpy as np
from sklearn.linear_model import RidgeCV
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp import load, splits, HERE

x, ages = load()
X = x.flatten(1).numpy()
rows = []
for sampling in ["balanced", "natural"]:
    for n in [100, 300, 1000]:
        for seed in range(5):
            tr, te = splits(ages, n, seed, sampling=sampling)
            yt = ages[te]
            r = {"sampling": sampling, "n": n, "seed": seed,
                 "const_train_mean": float(np.abs(ages[tr].mean() - yt).mean()),
                 "const_train_median": float(np.abs(np.median(ages[tr]) - yt).mean())}
            rid = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(X[tr], ages[tr])
            r["ridge_pixels"] = float(np.abs(rid.predict(X[te]) - yt).mean())
            rows.append(r)
            print(json.dumps(r), flush=True)
pool_med = float(np.abs(np.median(ages) - ages[te]).mean())
print("oracle const (whole-dataset median) MAE:", pool_med)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({"rows": rows, "const_dataset_median": pool_med}, open(os.path.join(HERE, "results", "baselines.json"), "w"))
