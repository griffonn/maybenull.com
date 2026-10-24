# Muon: Newton-Schulz step count vs exact and damped polar

Companion code for the maybenull.com post "Muon's Newton-Schulz steps, counted".
Tests the claims of Shulgin et al., *A Quadratic Lens on Muon's Polar Update*
(arXiv:2610.30280) in actual training.

## What runs

`muon_ns.py` trains a 4-layer char-level GPT (d=128, 4 heads, context 64,
0.81M parameters) on tiny-shakespeare for 600 (or 2000) steps, batch 32.
The 2-D weights inside the blocks use Muon (Nesterov momentum 0.95, update
scaled by sqrt(max(1, rows/cols))); embeddings, head and LayerNorms use AdamW
(lr 3e-3). Learning rate is constant, then decays linearly to 0 over the last 30%.
Validation loss is the mean over 8 fixed batches of 32 x 64 characters from the
last 10% of the text.

Orthogonalizer arms (`--arm`):

| arm | what Muon applies to the Frobenius-normalized momentum |
|---|---|
| `ns<q>` | q steps of the quintic Newton-Schulz iteration, coefficients (3.4445, -4.7750, 2.0315), fp32 |
| `polar` | exact polar factor U V^T via SVD (fp64) |
| `damped<q>` | U diag(s / sqrt(s^2 + eps)) V^T with eps = l_q^-2, l_q = 3.4445^q (the paper's damped-polar model of q NS steps) |
| `adamw` | AdamW on everything (reference) |

## Rerun

```sh
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # also needs numpy
./run_all.sh          # ~80 min on 4 cores
```

Single run: `THREADS=1 python3 muon_ns.py --arm ns5 --lr 0.02 --seed 0 --steps 600 --out results/x`
(~2 min alone; ~0.2 s/step). With `THREADS=1` this reproduces the shipped JSON
exactly on the hardware below (checked for `ns5`, seed 0: 1.6804); more threads
change the reduction order and move the final loss by a few thousandths.

## Files

- `filter_curves.py`: the scalar filters (NS for q steps, damped polar) -> `results/filter_curves.json`
- `run_sweep.sh`: LR sweep, seed 0 -> `results/sweep/`
- `run_seeds.sh`: seeds 1-4 at each arm's best LR, and 2000-step runs -> `results/seeds/`, `results/long/`
- `analyze.py` -> `results/summary.json` (per-arm best LR, per-seed final losses, curves)
- `spectra_share.py` -> `results/spectra_share.json` (share of momentum singular values below 1/l_5)
- `chart_data.py` -> `figs/*.json` (exactly what the post's charts plot)
- `results/gate/`: the first three pilot runs used to check that tiny singular values exist at this scale

Each results JSON holds the arguments, the validation curve, wall-clock and
optimizer time, and (with `--spectra`) the singular values of every Muon
momentum buffer at the end of training.

## Hardware and wall-clock

4-core Intel Xeon @ 2.80 GHz, no GPU, 15 GB RAM, torch 2.14.1+cpu, Python 3.11.
Sweep 37 min, seeds + long runs 41 min, 4 single-threaded runs at a time.
Timing columns (`opt_time`) are from a shared CPU and only roughly comparable.
