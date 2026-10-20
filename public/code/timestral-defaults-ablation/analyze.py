"""Print every table in the post from results/*.pkl."""
import glob, os, pickle
import numpy as np
from scipy.stats import spearmanr
from common import split, pinball_per_series, combine, MEDIAN

bench = pickle.load(open("bench.pkl", "rb"))
prep = {o: {n: [split(s, d["H"], d["m"]) for s in d["series"]] for n, d in bench[o].items()} for o in bench}
rng = np.random.default_rng(0)
BOOT = {o: [{n: rng.integers(0, len(v), len(v)) for n, v in prep[o].items()} for _ in range(1000)] for o in prep}


def load(model, ctx, origin, window=0):
    p = f"results/{model}_{ctx}_{origin}_w{window}.pkl"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def forecasts(model, ctx, origin, mode, window=0):
    r = load(model, ctx, origin, window)
    if r is None:
        return None
    out = {}
    for n, v in r.items():
        nonneg = np.array([x[4] for x in prep[origin][n]])
        out[n] = combine(v["qp"], v["qn"] if v["qn"] is not None else -v["qp"][..., ::-1], nonneg, mode)
    return out


def per_ds(Qs, origin):
    """Per dataset: (model loss, seasonal-naive loss) per series."""
    out = {}
    for n, rows in prep[origin].items():
        Y = np.stack([x[1] for x in rows]); SN = np.stack([x[3] for x in rows])
        num, _ = pinball_per_series(Qs[n], Y)
        base, _ = pinball_per_series(np.repeat(SN[..., None], 21, -1), Y)
        out[n] = (num, base)
    return out


def skill(Qs, origin, idx=None):
    """Geometric mean over datasets of WQL / seasonal-naive WQL (lower is better)."""
    logs = []
    for n, (num, base) in per_ds(Qs, origin).items():
        i = idx[n] if idx else slice(None)
        logs.append(np.log(num[i].sum() / base[i].sum()))
    return float(np.exp(np.mean(logs)))


def ratio_ci(Qa, Qb, origin):
    point = skill(Qa, origin) / skill(Qb, origin)
    d = [np.log(skill(Qa, origin, ix) / skill(Qb, origin, ix)) for ix in BOOT[origin]]
    lo, hi = np.exp(np.percentile(d, [2.5, 97.5]))
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


print("== Finding 1: mirror (default) vs single pass, full context")
for o in ("A", "B"):
    for m in ("5M", "28M", "209M"):
        a, b = forecasts(m, 32768, o, "sym"), forecasts(m, 32768, o, "raw")
        if a: print(f"origin {o} {m}: {ratio_ci(a, b, o)}")

print("\n== Finding 1: MSE decomposition, 209M full context, origin A")
r = load("209M", 32768, "A")
for n, v in r.items():
    rows = prep["A"][n]; sc = np.array([x[2] for x in rows])[:, None]
    nonneg = np.array([x[4] for x in rows]); Y = np.stack([x[1] for x in rows]) / sc
    fa = combine(v["qp"], v["qn"], nonneg, "raw")[..., MEDIAN] / sc
    fb = combine(v["qp"], v["qn"], nonneg, "mirror")[..., MEDIAN] / sc
    mse = lambda f: np.mean((f - Y) ** 2)
    corr = np.corrcoef((fa - Y).ravel(), (fb - Y).ravel())[0, 1]
    print(f"{n:26s} fwd {mse(fa):.3f} mirror {mse(fb):.3f} avg {mse((fa + fb) / 2):.3f} "
          f"gain {np.mean((fa - fb) ** 2) / 4:.3f} corr {corr:.3f}")

print("\n== Finding 1: per-dataset WQL/seasonal-naive, full context, origin A (fwd | default | mirror only)")
for m in ("5M", "28M", "209M"):
    ds = {k: per_ds(forecasts(m, 32768, "A", k), "A") for k in ("raw", "sym", "mirror")}
    for n in prep["A"]:
        print(f"{m:4s} {n:26s} " + " | ".join(f"{ds[k][n][0].sum() / ds[k][n][1].sum():.3f}" for k in ds))

print("\n== Finding 2: context sweep, single pass, origin A")
for m in ("5M", "28M", "209M"):
    print(m, "  ".join(f"{c}: {skill(forecasts(m, c, 'A', 'raw'), 'A'):.3f}" for c in (512, 2048, 8192, 32768)))
print("\n== Finding 2: ctx 2048 single pass vs full default")
for o in ("A", "B"):
    for m in ("5M", "28M", "209M"):
        a, b = forecasts(m, 2048, o, "raw"), forecasts(m, 32768, o, "sym")
        if a and b: print(f"origin {o} {m}: {ratio_ci(a, b, o)}")

print("\n== Finding 2: windowed-normalization intervention, origin A")
for m in ("5M", "209M"):
    w = forecasts(m, 32768, "A", "raw", window=2048)
    print(f"{m}: ctx2048 {skill(forecasts(m, 2048, 'A', 'raw'), 'A'):.3f}  ctx32768 "
          f"{skill(forecasts(m, 32768, 'A', 'raw'), 'A'):.3f}  ctx32768+window2048 {skill(w, 'A'):.3f}")
    full, short = per_ds(forecasts(m, 32768, "A", "raw"), "A"), per_ds(forecasts(m, 2048, "A", "raw"), "A")
    z, deg = [], []
    for n, d in bench["A"].items():
        for i, s in enumerate(d["series"]):
            h = s[:-d["H"]][-32768:]
            if len(h) < 8192: continue
            z.append(abs(np.nanmean(h[-2048:]) - np.nanmean(h)) / np.nanstd(h))
            deg.append(np.log((full[n][0][i] + 1e-9) / (short[n][0][i] + 1e-9)))
    rho, p = spearmanr(z, deg)
    print(f"   drift vs long-context penalty over {len(z)} series: spearman {rho:.2f} (p={p:.1e})")

print("\n== Finding 3: model size, ctx 2048 single pass")
for o in ("A", "B"):
    s5 = forecasts("5M", 2048, o, "raw")
    for m in ("28M", "209M"):
        q = forecasts(m, 2048, o, "raw")
        if q: print(f"origin {o} {m} vs 5M: {ratio_ci(q, s5, o)}")
    print(f"origin {o} 5M@2048 vs 209M default: {ratio_ci(s5, forecasts('209M', 32768, o, 'sym'), o)}")
