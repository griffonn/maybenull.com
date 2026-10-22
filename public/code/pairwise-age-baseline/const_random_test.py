"""Robustness check for the split: the repo's PairwiseUTKFace draws its 200 test faces at random
(not evenly spaced by age). Constant-prediction MAE over 20 such draws, with decade-balanced
training sets of N=100 drawn from the rest."""
import json, os
import numpy as np
from exp import load, HERE

_, ages = load()
maes = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    test = rng.choice(len(ages), 200, replace=False)
    pool = np.setdiff1d(np.arange(len(ages)), test)
    train = np.concatenate([rng.choice(b, 10, replace=len(b) < 10) for b in
                            (pool[np.minimum(ages[pool] // 10, 9) == d] for d in range(10))])
    maes.append(float(np.abs(ages[train].mean() - ages[test]).mean()))
print("constant (train mean) MAE over 20 random test draws: %.1f ± %.1f (range %.1f–%.1f)"
      % (np.mean(maes), np.std(maes, ddof=1), min(maes), max(maes)))
json.dump({"maes": maes}, open(os.path.join(HERE, "results", "const_random_test.json"), "w"))
