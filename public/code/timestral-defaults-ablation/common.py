"""Shared settings and metrics for the Timestral ablation."""
import numpy as np

TIMESTRAL_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
DATASETS_REVISION = "eeecad0b82a8c237e212ce6f8d1abecb513e2cec"  # autogluon/chronos_datasets

# name: (value column, horizon H, season m, series to sample, minimum length)
SPEC = {
    "electricity_15min":        ("consumption_kW", 96, 96, 32, 40000),
    "monash_traffic":           ("target",         48, 24, 32, 0),
    "ercot":                    ("target",         48, 24, 8,  0),
    "monash_pedestrian_counts": ("target",         48, 24, 32, 10000),
    "monash_kdd_cup_2018":      ("target",         48, 24, 32, 0),
    "solar_1h":                 ("power_mw",       48, 24, 32, 0),
    "m5":                       ("target",         28, 7,  32, 1800),
    "exchange_rate":            ("target",         30, 1,  8,  0),
    "taxi_30min":               ("target",         48, 48, 32, 0),
}

# Origin A: the forecast window is the last H steps of each series.
# Origin B: the same series with the last 7*m steps dropped first (30 steps when m=1), so the
# window ends one week earlier for the hourly / 15-min / 30-min datasets, 49 days earlier for M5
# and 30 days earlier for exchange rates.
def origin_offset(origin, m):
    if origin == "A":
        return 0
    return 7 * m if m > 1 else 30

LEVELS = np.array([0.01, 0.05] + [round(0.1 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99])
WQL_IDX = [i for i, l in enumerate(LEVELS) if abs(l * 10 - round(l * 10)) < 1e-9 and 0.05 < l < 0.95]  # 0.1 .. 0.9
MEDIAN = 10


def split(series, H, m):
    """History, truth, seasonal-naive scale and seasonal-naive forecast for one series."""
    hist, truth = series[:-H], series[-H:]
    scale = np.nanmean(np.abs(hist[m:] - hist[:-m]))
    filled = hist.copy()
    idx = np.where(np.isfinite(filled), np.arange(len(filled)), 0)
    filled = filled[np.maximum.accumulate(idx)]           # forward-fill
    filled = np.where(np.isfinite(filled), filled, np.nanmean(hist))
    snaive = np.resize(filled[-m:], H)
    return hist, truth, scale, snaive, bool(np.nanmin(hist) >= 0)


def pinball_per_series(Q, Y):
    """Per-series weighted-quantile-loss numerator (averaged over 0.1..0.9) and denominator."""
    num = np.zeros(len(Y))
    for j in WQL_IDX:
        q = LEVELS[j]
        e = Y - Q[..., j]
        num += np.nansum(np.maximum(q * e, (q - 1) * e), 1)
    return 2 * num / len(WQL_IDX), np.nansum(np.abs(Y), 1)


def combine(qp, qn, nonneg, mode):
    """Rebuild what TimestralForecaster.forecast returns from the two raw passes.

    qp: forward pass on context; qn: pass on -context.  Mirrors forecaster.py L220-L230.
    """
    mirrored = (-qn)[..., ::-1]
    Q = {"raw": qp, "sym": 0.5 * (qp + mirrored), "mirror": mirrored}[mode]
    return np.where(nonneg[:, None, None], np.maximum(Q, 0), Q)
