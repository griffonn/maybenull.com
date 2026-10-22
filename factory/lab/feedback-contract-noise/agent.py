"""Scripted hyperparameter hill-climber against three validation-feedback rules.

The "agent" proposes one local tweak of the current champion's config per
submission, trains a HistGradientBoostingClassifier, and submits its
validation predictions. The harness decides what to release:

  exact   - the exact validation error; champion = any candidate that improves it
  pse     - Algorithm 1 of arXiv 2610.28798 (= Blum & Hardt's parameter-free
            Ladder): accept only if error < champion - std(paired loss diff)/sqrt(n)
  ladder  - fixed-margin Ladder: accept only if error < champion - eta

Task 'signal' has learnable labels; task 'null' has labels drawn independently
of the features, so every validation gain over 0.5 is selection noise.
No patience stop here: we run the full budget and apply patience afterwards.

Two searchers:
  tweak     - each submission is one local tweak of the champion's config
  ensemble  - each submission is the champion ensemble plus one new member
              (champion's base config, one tweak, fresh random_state); hard
              predictions from the averaged probabilities. The everyday
              "add another model to the blend if validation improves".

usage: python agent.py TASK RULE SEED [K] [SEARCHER]
"""
import json, sys, time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier

N_TRAIN, N_VAL, N_TEST = 20_000, 25_000, 50_000
ETA = 0.005

def data(task, seed):
    X, y = make_classification(n_samples=N_TRAIN + N_VAL + N_TEST, n_features=30,
                               n_informative=10, flip_y=0.3, random_state=1000 + seed)
    if task == "null":
        y = np.random.default_rng(2000 + seed).integers(0, 2, len(y))
    a, b = N_TRAIN, N_TRAIN + N_VAL
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])

START = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
             l2_regularization=0.0, max_iter=100, max_features=0.8, random_state=0)

def tweak(cfg, rng):
    c = dict(cfg)
    k = rng.choice(list(c))
    if k == "learning_rate":     c[k] = float(np.clip(c[k] * rng.choice([0.8, 1.25]), 0.01, 0.5))
    elif k == "max_leaf_nodes":  c[k] = int(np.clip(c[k] + rng.choice([-4, 4]), 4, 127))
    elif k == "min_samples_leaf":c[k] = int(np.clip(c[k] + rng.choice([-5, 5]), 1, 500))
    elif k == "l2_regularization": c[k] = float(np.clip(c[k] + rng.choice([-0.5, 0.5]), 0, 20))
    elif k == "max_iter":        c[k] = int(np.clip(c[k] + rng.choice([-10, 10]), 10, 300))
    elif k == "max_features":    c[k] = float(np.clip(c[k] + rng.choice([-0.1, 0.1]), 0.2, 1.0))
    elif k == "random_state":    c[k] = int(rng.integers(1, 10**6))
    return c, k

def main():
    task, rule, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    searcher = sys.argv[5] if len(sys.argv) > 5 else "tweak"
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data(task, seed)
    rng = np.random.default_rng(seed)
    n = len(yva)
    cache = {}
    def fit(cfg):
        key = tuple(sorted(cfg.items()))
        if key not in cache:
            m = HistGradientBoostingClassifier(early_stopping=False, **cfg).fit(Xtr, ytr)
            cache[key] = (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
        return cache[key]
    members = []                                   # ensemble searcher: champion's member probas

    champ_cfg, champ_pred, L, R, Rtest = None, None, None, np.inf, None
    log, t0 = [], time.time()
    for t in range(K):
        if champ_cfg is None:
            cfg, knob = START, "start"
        else:
            cfg, knob = tweak(champ_cfg, rng)
        if searcher == "ensemble" and champ_cfg is not None:
            cfg = dict(cfg, random_state=int(rng.integers(1, 10**6)))
            qv, qt = fit(cfg)
            k = len(members)
            sv = (sum(a for a, _ in members) + qv) / (k + 1)
            st = (sum(b for _, b in members) + qt) / (k + 1)
        else:
            qv, qt = fit(cfg); sv, st = qv, qt
        pv, pt = (sv > 0.5).astype(int), (st > 0.5).astype(int)
        Lc = (pv != yva).astype(float)
        r = Lc.mean()
        if champ_cfg is None:
            m_diff, s, ok = n, np.nan, True
        else:
            m_diff = int((pv != champ_pred).sum())
            s = float(np.std(Lc - L))           # population std, as written in Algorithm 1
            if rule == "exact":   ok = r < R
            elif rule == "pse":   ok = r < R - s / np.sqrt(n)
            elif rule == "ladder":ok = r < R - ETA
        z = (R - r) / (s / np.sqrt(n)) if champ_cfg is not None and s > 0 else np.nan
        if ok:
            if searcher == "ensemble":
                members.append((qv, qt)); cfg = champ_cfg or cfg   # base config stays put
            champ_cfg, champ_pred, L, R = cfg, pv, Lc, r
            Rtest = float((pt != yte).mean())
        log.append(dict(t=t + 1, knob=knob, m=m_diff, cand_val=float(r), z=float(z),
                        accepted=bool(ok), champ_val=float(R), champ_test=Rtest))
    out = dict(task=task, rule=rule, seed=seed, K=K, searcher=searcher, n_members=len(members), n_val=n, eta=ETA,
               wall_s=time.time() - t0, log=log)
    json.dump(out, open(f"results/agent_{searcher}_{task}_{rule}_{seed}.json", "w"))
    print(searcher, task, rule, seed, f"val {1-R:.4f} test {1-Rtest:.4f} accepts "
          f"{sum(e['accepted'] for e in log)} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
