"""Run both raw passes (context and -context) for one model / context length / origin.

    python run.py MODEL CTX ORIGIN [--window W]

--window W monkeypatches the scaler so loc/scale come from a trailing W-step window instead of
the cumulative history (scaler.py L14-L59 and the horizon frames in scaler.py L165-L210).
"""
import argparse, os, pickle, time
import numpy as np, torch
import timestral.scaler as S, timestral.forecaster as F
from timestral import TimestralForecaster

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("ctx", type=int); ap.add_argument("origin")
ap.add_argument("--window", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(4)

if a.window:
    W = a.window
    orig_frames = S.advancing_block_frames

    def windowed_stats(data, observation_mask, minimum_scale, stabilize_with_global, scale_factor_exponent=10.0):
        x = data.to(torch.float64); w = observation_mask.to(torch.float64)
        def rolling(v):
            c = torch.cumsum(v, -1); out = c.clone(); out[..., W:] = c[..., W:] - c[..., :-W]; return out
        n, s1, s2 = rolling(w), rolling(w * x), rolling(w * x * x)
        mean = s1 / n.clamp_min(1.0)
        var = ((s2 - n * mean * mean) / (n - 1.0).clamp_min(1.0)).clamp_min(0.0)
        return mean.to(data.dtype), var.sqrt().clamp(min=minimum_scale).to(data.dtype)

    S.compute_causal_statistics = windowed_stats
    F.advancing_block_frames = lambda ctx, m, sm, ms: orig_frames(ctx[..., -W:], m[..., -W:], sm, ms)

bench = pickle.load(open("bench.pkl", "rb"))[a.origin]
f = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{a.model}")
res = {}
for name, d in bench.items():
    H = d["H"]
    hists = [s[:-H][-a.ctx:] for s in d["series"]]
    L = max(len(h) for h in hists)
    X = np.full((len(hists), L), np.nan, dtype=np.float32)   # left-pad with NaN = missing
    for i, h in enumerate(hists):
        X[i, L - len(h):] = h
    qp, qn, secs = [], [], 0.0
    for b in range(0, len(X), 8):
        inp, mask = f._as_context(X[b:b + 8], None)
        t = time.perf_counter()
        with torch.inference_mode():
            qp.append(f._forecast_raw(inp, mask, H, past_covariates=None, known_covariates=None)[:, 0].numpy())
            if not a.window:
                qn.append(f._forecast_raw(-inp, mask, H, past_covariates=None, known_covariates=None)[:, 0].numpy())
        secs += time.perf_counter() - t
    res[name] = dict(qp=np.concatenate(qp), qn=np.concatenate(qn) if qn else None, secs=secs)
    print(f"{a.model} ctx={a.ctx} origin={a.origin} window={a.window} {name}: {secs:.1f}s", flush=True)
os.makedirs("results", exist_ok=True)
pickle.dump(res, open(f"results/{a.model}_{a.ctx}_{a.origin}_w{a.window}.pkl", "wb"))
