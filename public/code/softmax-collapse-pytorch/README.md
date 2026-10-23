# Softmax Collapse in PyTorch

Experiment code for the maybenull.com post on
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642) (Barron, Gnanapragasam, Tschannen, 2026).
The paper's code is JAX; `ce.py` is our PyTorch port of its hard-target cross-entropy.

Questions: where does PyTorch's own `F.cross_entropy` zero the target gradient (fp32, bf16, fp16, autocast)?
On the paper's grokking task, does the fix beat one-line knobs (Adam eps 1e-7, label smoothing, fp64 loss)?
Does an ordinary classifier ever cross the cliff? What does the fix cost?

## Files

| File | What |
|---|---|
| `ce.py` | `fixed_ce` (direct port, autograd), `fixed_ce_fn` (same math as a custom `autograd.Function`), `margin` (fp64). |
| `cliff.py` | Sweeps the logit margin m from 0 to 100 (K=32) for `F.cross_entropy`, `log_softmax+nll_loss`, fixed CE, in fp32/bf16/fp16/fp64 and bf16 autocast. `results/cliff.csv`. |
| `cliff_k.py` | Cliff location (0.01 steps) of `F.cross_entropy` in fp32/bf16/fp16/fp64 vs number of classes K and how the non-target mass is spread. `results/cliff_k.csv`. |
| `grok.py` | One grokking run, a+b mod 31. Arms listed in `ARMS`. `results/grok/ARM_sSEED.json` (every 10 steps: exact fp64 train CE, mean/min margin, fraction of examples whose fp32 softmax target prob is exactly 1, train/test acc). |
| `run_grok.sh` | 10 arms x 5 seeds, 4 parallel, 1 thread each. |
| `run_lr1e-3.sh` | Same model at the paper's lr 1e-3 (2 seeds, std and fixed): why we use 3e-3. |
| `digits.py`, `run_digits.sh` | sklearn's bundled 8x8 digits, MLP 64-256-256-10, full-batch AdamW, 20k steps, std vs fixed, wd 0 and 0.01, 3 seeds. |
| `bench_speed.py` | Forward+backward ms: `F.cross_entropy` vs fixed CE (autograd and `autograd.Function`), vocab 1k/32k/128k, fp32/bf16, 4 threads, 3 rounds. `results/speed.csv`, ratio ranges in `results/speed_summary.csv`. |
| `analyze.py` | `results/summary.json` + the markdown table. Spike = exact train CE > 10x its minimum over the preceding 500 steps (consecutive points count once). Dip = largest drop of test acc below its running max after first reaching 90%. |
| `chart_data.py` | `figs/cliff.json`, `figs/grok-curves.json` for the post's charts. |
| `run_all.sh` | Everything in order. |

## Grokking setup

1-layer transformer: token + learned position embeddings (inputs `a b =`), 4-head attention, ReLU MLP 64→256→64,
residuals, no LayerNorm, unembedding read at `=`. Width 64, p=31, 50% random train split (by seed), full-batch AdamW,
betas (0.9, 0.98), weight decay 1.0, eps 1e-8 unless the arm says otherwise, 10,000 steps, PyTorch default init.
**lr 3e-3**, not the paper's 1e-3: at 1e-3 our model sits at 52–72% test accuracy at step 10k (`run_lr1e-3.sh`).

Arms: `std_fp32`, `fixed_fp32`, `std_fp32_eps1e-7`, `std_fp32_ls0.01` (label smoothing 0.01), `std_fp32_loss64`
(logits cast to fp64 before `F.cross_entropy`), `std_autocast_bf16` (fp32 weights, `torch.autocast` bf16; autocast runs
CE in fp32), `std_bf16logits` / `fixed_bf16logits` (fp32 weights, autocast forward, loss computed on the bf16 logits),
`std_bf16` / `fixed_bf16` (whole model and AdamW state in bf16).

## Rerun

```
python -m venv venv && venv/bin/pip install torch==2.14.1 numpy scikit-learn
PY=venv/bin/python ./run_all.sh
```

Hardware: 4-core Intel Xeon @ 2.10GHz, no GPU, Python 3.11, torch 2.14.1+cpu (`results/hardware.txt`).
Wall-clock: about 50 minutes for `run_all.sh`. Results are deterministic per seed on this machine; other CPUs or
thread counts can change low bits and therefore individual trajectories.
