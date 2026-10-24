"""From saved Nesterov-momentum spectra (end of training, Frobenius-normalized):
fraction of singular values below 1/l_5, and the share of each update's squared
Frobenius norm those directions get under exact polar vs 5 quintic NS steps."""
import glob, json, os, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
A, B, C = 3.4445, -4.7750, 2.0315
thr = A ** -5

def phi(s, q):
    for _ in range(q):
        s = A * s + B * s**3 + C * s**5
    return s

out = {}
for arm in ["ns5", "polar"]:
    files = sorted(glob.glob(os.path.join(HERE, "results/seeds", f"{arm}_lr0.02_s*_n600.json")))
    per = collections.defaultdict(list); tot = dict(frac=[], polar_share=[], ns5_share=[])
    for f in files:
        sp = json.load(open(f))["momentum_spectra"]
        all_s = []
        for name, s in sp.items():
            s = np.array(s); kind = name.split(".")[2]
            per[kind].append(float((s < thr).mean())); all_s.append(s)
        s = np.concatenate(all_s); small = s < thr
        tot["frac"].append(float(small.mean()))
        tot["polar_share"].append(float(small.sum() / len(s)))           # polar: every direction gets 1
        p = phi(s, 5) ** 2; tot["ns5_share"].append(float(p[small].sum() / p.sum()))
    out[arm] = dict(n_runs=len(files), by_matrix={k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k, v in per.items()},
                    overall={k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k, v in tot.items()})
    print(arm, json.dumps(out[arm], indent=1))
json.dump(out, open(os.path.join(HERE, "results", "spectra_share.json"), "w"), indent=1)
