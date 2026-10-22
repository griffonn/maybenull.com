"""Pure-noise simulations of the release rules, no models involved.

Labels are random bits, so every candidate has true accuracy 0.5 and any
validation gain is selection noise. Two searchers:

  local   - each submission is the champion with m random prediction bits flipped
  boost   - Blum & Hardt's boosting attack: submit k independent random vectors,
            then submit the majority vote of the ones the harness said were good

usage: python sim.py   (writes results/sim_local.csv, results/sim_boost.csv; ~1 min)
"""
import csv
import numpy as np

N, K, SEEDS, ETA = 25_000, 1000, 5, 0.005
MS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

def accept(rule, r, R, Lc, L):
    if rule == "exact":  return r < R
    if rule == "pse":    return r < R - np.std(Lc - L) / np.sqrt(len(L))
    if rule == "pse1":   return r < R - np.std(Lc - L, ddof=1) / np.sqrt(len(L))
    if rule == "ladder": return r < R - ETA

def local(rule, m, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    champ = rng.integers(0, 2, N)
    L = (champ != y).astype(float); R = L.mean()
    y_test = rng.integers(0, 2, N)            # fresh labels: true accuracy stays 0.5
    curve, acc = [], 0
    for t in range(K):
        cand = champ.copy(); cand[rng.choice(N, m, replace=False)] ^= 1
        Lc = (cand != y).astype(float); r = Lc.mean()
        if accept(rule, r, R, Lc, L):
            champ, L, R = cand, Lc, r; acc += 1
        curve.append(0.5 - R)
    return curve, acc, 0.5 - (champ != y_test).mean()

def boost(rule, seed, k=K):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    y_test = rng.integers(0, 2, N)
    L, R, good = np.zeros(N), np.inf, []
    for t in range(k):
        v = rng.integers(0, 2, N)
        Lc = (v != y).astype(float); r = Lc.mean()
        if rule == "exact":
            ok = r < 0.5                           # exact score tells you which side of 0.5
        else:
            ok = (R == np.inf) or accept(rule, r, R, Lc, L)
            if ok: L, R = Lc, r
        if ok: good.append(v)
    vote = (np.mean(good, axis=0) > 0.5).astype(int) if good else rng.integers(0, 2, N)
    return 0.5 - (vote != y).mean(), 0.5 - (vote != y_test).mean(), len(good)

def first_step(m, trials=4000, seed=0):
    """Pass rate of the paired-SE rule when the champion has not been selected yet:
    fresh random champion, candidate = champion with m bits flipped."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        y = rng.integers(0, 2, N); c = rng.integers(0, 2, N)
        L = (c != y).astype(float)
        d = c.copy(); d[rng.choice(N, m, replace=False)] ^= 1
        Lc = (d != y).astype(float)
        hits += accept("pse", Lc.mean(), L.mean(), Lc, L)
    return hits / trials

if __name__ == "__main__":
    with open("results/sim_firststep.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["m", "pass_rate"])
        for m in [2, 10, 100, 1000, 10000]:
            w.writerow([m, first_step(m)])
    with open("results/sim_local.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rule", "m", "seed", "accepts", "val_gain_final", "test_gain_final", "curve"])
        for rule in ["exact", "pse", "pse1", "ladder"]:
            for m in MS:
                for s in range(SEEDS):
                    c, a, tg = local(rule, m, s)
                    w.writerow([rule, m, s, a, f"{c[-1]:.5f}", f"{tg:.5f}", " ".join(f"{x:.5f}" for x in c)])
                print(rule, m, flush=True)
    with open("results/sim_boost.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rule", "seed", "val_gain", "test_gain", "n_used"])
        for rule in ["exact", "pse", "ladder"]:
            for s in range(SEEDS):
                vg, tg, nu = boost(rule, s)
                w.writerow([rule, s, f"{vg:.5f}", f"{tg:.5f}", nu])
            print("boost", rule, flush=True)
