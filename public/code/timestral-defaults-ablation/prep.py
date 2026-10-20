"""Download the 9 datasets (pinned revision) and sample the evaluation series for both origins."""
import os, pickle, urllib.request
import numpy as np, pandas as pd
from common import SPEC, DATASETS_REVISION, origin_offset

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)
bench = {"A": {}, "B": {}}
for name, (col, H, m, n, minlen) in SPEC.items():
    path = f"data/{name}.parquet"
    if not os.path.exists(path):
        url = (f"https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/"
               f"{DATASETS_REVISION}/{name}/train-00000-of-00001.parquet")
        urllib.request.urlretrieve(url, path)
    ys = [np.asarray(v, dtype=np.float64) for v in pd.read_parquet(path, columns=[col])[col]]
    ok = [i for i, y in enumerate(ys)
          if len(y) >= max(minlen, H + 4 * m + 512) and np.isfinite(y[-H:]).all() and np.nanstd(y[-H - 2000:-H]) > 0]
    if name == "m5":        # keep items that actually sell
        ok = [i for i in ok if (ys[i][-400:] > 0).mean() > 0.3]
    if name == "solar_1h":  # drop panels that are dead in the last week
        ok = [i for i in ok if ys[i][-24 * 7:].max() > 0]
    pick = rng.choice(ok, size=min(n, len(ok)), replace=False)
    for origin in ("A", "B"):
        off = origin_offset(origin, m)
        series = [ys[i][: len(ys[i]) - off][-(32768 + H):] for i in pick]
        bench[origin][name] = dict(H=H, m=m, series=series)
    print(f"{name}: {len(ok)} eligible, sampled {len(pick)}")
pickle.dump(bench, open("bench.pkl", "wb"))
