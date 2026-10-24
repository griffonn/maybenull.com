"""Scalar filters phi(s) applied to Frobenius-normalized singular values:
Muon quintic NS with q steps, exact polar (phi=1), damped polar s/sqrt(s^2+l_q^-2)."""
import json, numpy as np
A, B, C = 3.4445, -4.7750, 2.0315
s = np.logspace(-5, 0, 1200)
out = {"s": s.tolist(), "ns": {}, "damped": {}, "l_q": {}}
for q in (1, 2, 3, 5, 8):
    x = s.copy()
    for _ in range(q):
        x = A * x + B * x**3 + C * x**5
    out["ns"][q] = x.tolist(); out["l_q"][q] = A**q
    out["damped"][q] = (s / np.sqrt(s * s + A ** (-2 * q))).tolist()
# where does each NS filter first reach 0.5?
for q in (1, 2, 3, 5, 8):
    x = np.array(out["ns"][q]); i = np.argmax(x >= 0.5)
    print(f"q={q} l_q={A**q:9.1f} 1/l_q={A**-q:.2e} NS reaches 0.5 at s={s[i]:.2e}; range on [1e-2,1]: {x[s>=1e-2].min():.3f}-{x[s>=1e-2].max():.3f}")
json.dump(out, open("results/filter_curves.json", "w"))
