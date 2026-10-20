"""Check that combine() over the two raw passes reproduces forecast() defaults exactly."""
import pickle, numpy as np, torch
from timestral import TimestralForecaster
from common import combine

f = TimestralForecaster.from_pretrained("mistralai/Timestral-5M")
d = pickle.load(open("bench.pkl", "rb"))["A"]["monash_traffic"]
x = np.stack([s[:-48][-2048:] for s in d["series"][:8]]).astype(np.float32)
lib = f.forecast(x, 48).quantiles[:, 0].numpy()
inp, mask = f._as_context(x, None)
with torch.inference_mode():
    p = f._forecast_raw(inp, mask, 48, past_covariates=None, known_covariates=None)[:, 0].numpy()
    n = f._forecast_raw(-inp, mask, 48, past_covariates=None, known_covariates=None)[:, 0].numpy()
mine = combine(p, n, np.array([True] * len(x)), "sym")
print("max abs diff vs forecast():", float(np.abs(lib - mine).max()))
