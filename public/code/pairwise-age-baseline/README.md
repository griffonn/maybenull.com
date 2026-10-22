# Pairwise vs. pointwise age regression at small N

Experiment code for the maybenull.com post on
[Pairwise Learning Increases Data Efficiency in Deep Neural Networks](https://arxiv.org/abs/2610.27669)
(code: [bainro/pairwise_learning](https://github.com/bainro/pairwise_learning) @ `33eae832`).

Question: the paper's headline small-data result is pairwise MAE 13.9 vs pointwise MSE 22.6 on
UTKFace with 100 training faces. How much of that gap is pairwise learning, and how much is the
baseline and the split?

## Files

| File | What |
|---|---|
| `prep.py` | Decodes UTKFace into a 64×64 grayscale cache (`data/utk64.npz`, not committed). Grayscale formula as in the paper's code. |
| `exp.py` | One training run. Arms: `pair` (2-channel pair input, cross-entropy on "which is older", the paper's max-entropy decode against the training images as references), `mse` `l1` (1-channel input, standardized age target), and `mse_raw` (MSE on raw age in years). Same backbone (the repo's ResNet-10-style `AgeRanker`, as its forward pass actually runs), AdamW 5e-4 / wd 1e-2, batch 32, horizontal flips, fixed step budget, no early stopping. |
| `baselines.py` | Constant predictors (training mean/median) and ridge regression on raw pixels, on the same splits. |
| `const_random_test.py` | Constant-predictor MAE when the 200 test faces are drawn at random, as the repo does. |
| `analyze.py` | Markdown table + `results/summary.csv`. |
| `plot.py` | `figs/mae-vs-n.png`, `figs/scatter-n100.png`. |
| `run.sh` | Main grid: 3,000 steps, N ∈ {100, 300, 1000}. ~80 min on 4 cores. |
| `run_long.sh` | 10,000-step runs at N=100, both arms, 3 seeds. ~25 min. |
| `run_raw.sh` | `mse_raw` at N=100, 5 seeds. ~8 min. |
| `results/` | One JSON per run (args, learning curve, per-image predictions), `baselines.json`, `summary.csv`, `hardware.txt`. |

## Splits

- **Test:** 200 images at evenly spaced positions of the age-sorted dataset (the paper: test points
  "sampled evenly throughout the whole sorted list"). Fixed across seeds and arms.
- **Train (`balanced`, default):** the repo's `PairwiseUTKFace` scheme, N/10 images per age decade
  (0–9, …, 90+), drawn with replacement where a decade is short. This is the split the repo's code
  produces; the paper doesn't state the split used for its N=100 figure.
- **Train (`natural`):** N images drawn uniformly from the non-test pool.

## Differences from the paper

64×64 inputs instead of 192×192 (CPU budget), no 8× test-time augmentation for any arm, fixed
3,000 steps with the final checkpoint reported (plus 10,000-step runs of both arms at N=100), and the
"best checkpoint on test" number reported separately to show what test-set early stopping would add.

## Rerun

```sh
pip install torch numpy scipy scikit-learn matplotlib pillow
# UTKFace ("UTKFace" folder of jangedoo/utkface-new on Kaggle, 23,708 images)
python3 prep.py /path/to/UTKFace data/utk64.npz     # ~30 s
python3 baselines.py                                # ~10 s idle
python3 const_random_test.py                        # constant MAE on 20 random test draws
./run.sh && ./run_long.sh && ./run_raw.sh           # ~2 h on 4 cores
python3 analyze.py && python3 plot.py
```

Total experiment wall-clock for the published numbers: ~1 h 56 min on 4 cores (22:54–00:50 UTC).
