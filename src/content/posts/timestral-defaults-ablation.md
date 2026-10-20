---
title: "Half of Timestral's compute goes to a mirror image"
description: Ablating Mistral's new time-series models on 9 real datasets. The default second pass doubles the cost for no measurable gain, and the 32k context mostly buys latency.
date: 2026-10-20
draft: false
---

Mistral released [Timestral](https://mistral.ai/news/timestral) yesterday. It's a family
of probabilistic time-series forecasters with 5M, 28M, 123M and 209M parameters, open
weights, and a 32,768-step context. The headline claims are leaderboard ranks (#1 among
zero-shot models on GIFT-Eval, TIME and fev-bench) and CPU latencies. Per the
[repo README](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/README.md#L132-L133),
those benchmarks ran with the forecaster's defaults and the full history (up to 32,768
steps) as context.

I wanted to know what those defaults cost and what they buy, so I read the
inference code and ran ablations on my own CPU. Everything below is my measurement
unless it's attributed to someone else. All code links point to Timestral commit
[`a534bdb`](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff),
and the full reproduction code is [linked at the end](#reproduce-it).

## Two things the defaults do

**1. Every forecast runs the model twice.** `forecast()` has
[`symmetric_averaging=True`](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py#L170-L171)
by default. The
[core of it](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py#L220-L230)
is a few lines:

```python
quantiles = self._forecast_raw(inputs, mask, prediction_length, **covariates)
if symmetric_averaging:
    mirrored = self._forecast_raw(-inputs, mask, prediction_length, **mirrored_covariates)
    quantiles = 0.5 * (quantiles + (-mirrored).flip(-1))
```

It forecasts your series, then forecasts its *negation* (covariates are negated too). It
flips that second forecast back: negate the values and reverse the quantile axis, so the
mirror's 1% quantile becomes the original's 99% quantile. Then it averages the two. The
[module docstring](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py#L8-L14)
calls this, together with clipping non-negative series at zero, "clampsym" test-time
augmentation. It enforces a symmetry the network only approximately learned:
f(−x) = −f(x).

**2. It reads up to 32,768 steps of history.** The forecaster keeps the
[most recent `max_context` steps](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py#L373-L375),
which defaults to the trained context length of 32,768. Patches are 32 steps
([`patch_size` in the model config](https://huggingface.co/mistralai/Timestral-209M/blob/9874d252ab0aae16f7f7c999f98f98d6ad7d8365/config.json)),
so a full context is 1,024 tokens per pass, and attention cost grows with the token
count.

Here is batch-1 latency on a 4-vCPU Xeon (2.1 GHz), forecasting 48 steps (best of 3 runs;
expect about ±15% between runs):

| model | ctx 2048 | ctx 2048 + mirror | ctx 8192 | ctx 32768 | ctx 32768 + mirror (default, long series) |
|---|---|---|---|---|---|
| 5M | 11 ms | 24 ms | 31 ms | 97 ms | 189 ms |
| 28M | 26 ms | 49 ms | 54 ms | 276 ms | 585 ms |
| 123M | 74 ms | 164 ms | 193 ms | 752 ms | 1,582 ms |
| 209M | 132 ms | 320 ms | 347 ms | 1,183 ms | **2,512 ms** |

Between the cheapest and the default setting, 209M differs by **19×**. Mistral's quoted
140 ms for 209M is for three weeks of 15-minute data (2,016 steps) on 8 threads, which is
in line with my 320 ms on 4 threads with the mirror. If you hand it a long history, which
is what the defaults invite, you pay roughly 8–20× that.

So the question is whether that compute shows up in accuracy.

## Setup

- **Data:** 9 datasets from the [Chronos benchmark collection](https://huggingface.co/datasets/autogluon/chronos_datasets):
  15-min electricity, hourly road traffic, ERCOT load, pedestrian counts, KDD-2018 air
  quality, hourly solar, M5 retail (daily, intermittent), exchange rates (daily) and
  30-min NYC taxi. I sampled 32 series per dataset (8 for ERCOT and exchange rate,
  which only have 8). Horizons are 96 steps for 15-min data, 48 for hourly and 30-min,
  28 for M5 and 30 for exchange rates.
- **Two forecast windows per series.** A single forecast window can be lucky, so I
  forecast each series twice:
  - **latest:** predict the last H steps of the data, from everything before them;
  - **earlier:** drop the last 7 seasons first, then do the same. That moves the window
    one week back for the hourly, 30-min and 15-min data, 49 days back for M5 and 30
    days back for exchange rates.

  The two windows give an independent check on each result. Where only the latest
  window is shown, I only ran that configuration there.
- **Metric:** weighted quantile loss over the 0.1…0.9 quantiles. I divide it by the
  seasonal-naive baseline on the same windows and take the geometric mean across
  datasets, fev-bench style. **Lower is better**: 0.55 means 45% less loss than repeating
  the last season (last day for hourly data, last week for daily data).
- **Uncertainty:** 95% intervals come from 1,000 bootstrap resamples of series
  within each dataset.
- **Trick:** I call the model's two raw passes myself and combine them exactly as
  `forecast()` does. I checked that my reconstruction matches the library's default
  output bit for bit (max abs diff 0.0). That way every config gets "with mirror" and
  "without mirror" from the same runs.

## Finding 1: the mirror pass does nothing measurable

Here is the loss ratio of the default (with mirror) to a single pass, at full context.
Below 1 means the mirror helps:

| model | latest window | earlier window |
|---|---|---|
| 5M | 1.004 [0.998, 1.012] | 0.992 [0.986, 1.000] |
| 28M | 0.997 [0.989, 1.003] | not run |
| 209M | 0.998 [0.979, 1.017] | 1.000 [0.995, 1.007] |

Every interval contains 1.0, and every point estimate is within 0.8%. You can pass
`symmetric_averaging=False` and halve your latency.

That isn't because the two passes agree. Per series, the forward and mirrored
medians differ by 0.05–0.35 seasonal-MAE units. Per dataset, the mirror alone beats
the forward pass by up to 18% (ERCOT) or loses by up to 19% (pedestrians, traffic).
The reason averaging doesn't help is a one-line identity. For squared error, with
forward forecast *a* and mirrored forecast *b*:

```
MSE((a+b)/2) = (MSE(a) + MSE(b)) / 2  −  mean((a−b)²) / 4
                 └── average of the two ──┘    └─ ensemble gain ─┘
```

The gain is a quarter of the squared disagreement. The disagreement is small next to
the errors, because the two passes make almost the same mistakes:

| dataset (209M, full ctx, latest window) | MSE fwd | MSE mirror | MSE avg | gain (a−b)²/4 | error correlation |
|---|---|---|---|---|---|
| electricity 15min | 5.456 | 5.532 | 5.482 | 0.012 | 0.996 |
| traffic | 0.518 | 0.552 | 0.527 | 0.008 | 0.966 |
| pedestrians | 0.319 | 0.573 | 0.419 | 0.027 | 0.911 |
| solar | 3.045 | 2.913 | 2.906 | 0.073 | 0.949 |
| M5 | 1.888 | 1.905 | 1.895 | 0.001 | 0.998 |
| exchange rate | 23.61 | 21.10 | 22.32 | 0.037 | 0.997 |

(MSE of the median, in units of each series' seasonal-naive MAE.)

With error correlations of 0.91–0.998, the "ensemble" is really one model asked twice.
What's left is the *prior*. It helps where the data actually is roughly sign-symmetric:
exchange rates, close to a random walk, improve 2–4%. It hurts where it isn't.
Pedestrian counts have a hard floor at zero, and negating them gives a series with a
hard *ceiling* at zero, a shape real data rarely has. There the 209M mirror pass has
80% higher error and drags the average down. Across a mixed benchmark the two effects
cancel. On your own data the effect is predictable: off for counts and demand, maybe
on for price-like series.

### This isn't new, and nobody else found much either

Timestral didn't invent this, and its code doesn't credit anyone. The same pair of
defaults, `force_flip_invariance=True` and `infer_is_positive=True`, is in Google's
TimesFM 2.5, in the very
[first commit of its 2.0.0 package](https://github.com/google-research/timesfm/blob/7d8f3d971d66daa57b74be01bb1021c1d99bb685/src/timesfm/configs.py#L31-L32)
(September 2025). The
[implementation](https://github.com/google-research/timesfm/blob/7d8f3d971d66daa57b74be01bb1021c1d99bb685/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py#L359-L374)
decodes `-inputs`, reverses the quantiles, and averages.

I couldn't find a paper that proposes the trick. Two describe or measure it, and both
put the gain at about half a percent:

- **[Reverso](https://arxiv.org/abs/2602.17634)** (Fu et al., Feb 2026) uses it at
  inference (§3.3). It writes the combination as (f(x) − f(−x))/2, which is the same
  average, since −f(−x) is the reflected forecast. Its ablation (Table 7, GIFT-Eval)
  moves overall MASE from 0.726 without the flip to 0.722 with it. Reverso attributes
  the idea to the original TimesFM paper ([Das et al., 2024](https://arxiv.org/abs/2310.10688)),
  but I searched that paper's full text and it never mentions flips or negation. The
  trick appears in TimesFM's code, not its paper.
- **["Foundation or Formula?"](https://arxiv.org/abs/2610.00589)** (Ebrahim et al.,
  Sep 2026) checks TimesFM-3 with and without these evaluator settings (§5.2, §6.3).
  M4 OWA moves from 0.862 to 0.857, and the authors report that no conclusion changes.

Three independent measurements, on three model families, put the mirror pass's effect in the
0–0.8% range, at 2× the inference cost.

## Finding 2: more context isn't better, and you can't recover it

Here is the context sweep (single pass, latest window):

| model | 512 | 2048 | 8192 | 32768 |
|---|---|---|---|---|
| 5M | 0.633 | **0.579** | 0.588 | 0.600 |
| 28M | 0.610 | **0.578** | 0.593 | 0.599 |
| 209M | 0.595 | 0.546 | **0.545** | 0.562 |

512 steps is clearly too short. Beyond about 2k, the curve is flat or gets worse. Here
are the paired comparisons of 2048 steps with a single pass against the full default
(below 1 means the cheap setting is more accurate):

| model | latest window | earlier window |
|---|---|---|
| 5M | 0.960 [0.943, 0.975] | 0.974 [0.957, 0.999] |
| 28M | 0.968 [0.952, 0.983] | not run |
| 209M | 0.974 [0.955, 0.990] | 1.018 [0.987, 1.051] |

For the small models, 2k context with no mirror is **better** in both windows, and
17–22× faster. For 209M it's a wash: 2.6% better in one window, 1.8% worse in the
other.

I expected a specific mechanism, and it turned out wrong, which is worth reporting.
Timestral
[normalizes each patch with the *cumulative* mean and standard deviation](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/scaler.py#L14-L59)
of everything before it, and then
[squashes with `asinh`](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/scaler.py#L88-L100).
Forecasts are
[denormalized in the frame of the last context patch](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/model.py#L227-L229),
so at 32k steps that frame is set by years of hourly history. If the recent level has
drifted from that, the recent dynamics land in the compressed tail of `asinh`. It's a
neat story, and the 5M data partly fits it: the loss from long context correlates with
how far the last 2k steps drift from the full-history mean (Spearman 0.28, p < 0.001).

So I tested it directly. I monkeypatched `compute_causal_statistics` so the full 32k
context is normalized with **trailing 2,048-step** statistics. I did the same for the
[horizon frames in `advancing_block_frames`](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/scaler.py#L165-L210).
Results (latest window):

| | ctx 2048 | ctx 32768 | ctx 32768, windowed normalization |
|---|---|---|---|
| 5M | 0.579 | 0.600 | 0.604 |
| 209M | 0.546 | 0.562 | 0.558 |

It recovers almost nothing, and for 209M the drift correlation is −0.06. So the
normalization frame isn't the cause. The extra degradation comes from what attention
does with the old patches, not from how they're scaled.

## Finding 3: 5M is most of the way there

At the same 2k single-pass setting, 28M is indistinguishable from 5M (ratio 0.999
[0.983, 1.013]). 209M is 5.6% better in the latest window and 7.1% better in the
earlier one. Against the full 209M default, 5M at 2k is 3–10% behind and about
**230× cheaper** (11 ms vs 2,512 ms).

## What I'd actually run

```python
f = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")   # or -5M
fc = f.forecast(history[-2048:], prediction_length=48,
                symmetric_averaging=False)      # keep clip_nonnegative=True
```

That's about 19× faster than the defaults on long histories. For 209M it's within a
few percent on accuracy in either direction, and for 5M and 28M it's slightly better.
If you need the last few percent, sweep the context length on your own data: the best
length differed by model and by window here, and no single value won everywhere.

## Caveats

- This is univariate only, with no covariates. Timestral's covariate attention is a big
  part of its pitch, and none of this tests it.
- These datasets are standard benchmark material, and Mistral post-trained on real
  data, so absolute skill may be flattered by overlap. The ablations compare settings
  within one model, which makes them less exposed to that.
- 32 series per dataset and two windows. The intervals reflect series sampling, not
  every source of variance.
- Latencies are from one 4-vCPU VM in fp32 PyTorch. A GPU changes the absolute
  numbers, but the mirror is still a second full forward pass.

## Reproduce it

Everything is in [`/code/timestral-defaults-ablation/`](https://github.com/griffonn/maybenull.com/tree/main/public/code/timestral-defaults-ablation)
([README](/code/timestral-defaults-ablation/README.md)). It pins the Timestral commit,
the dataset revision and the package versions. `run_all.sh` runs every experiment in
about 45 minutes on 4 vCPUs, and `analysis.txt` / `latency.txt` hold the exact output
behind the tables above. The scripts are:

- `prep.py` samples the series (seed 42) for both windows.
- `check_equivalence.py` verifies the two-pass reconstruction against `forecast()`.
- `run.py` runs the forward and mirrored passes, with `--window` for the normalization
  experiment.
- `analyze.py` prints every table.

*Sources: [Mistral's announcement](https://mistral.ai/news/timestral),
[mistralai/timestral](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff),
[google-research/timesfm](https://github.com/google-research/timesfm),
[Reverso](https://arxiv.org/abs/2602.17634), ["Foundation or Formula?"](https://arxiv.org/abs/2610.00589),
[Chronos datasets](https://huggingface.co/datasets/autogluon/chronos_datasets).
Benchmarks, analysis and conclusions are my own.*
