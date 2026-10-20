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
[repo README](https://github.com/mistralai/timestral), those benchmarks ran with the
forecaster's defaults and the full history as context.

I wanted to know what those defaults cost and what they buy, so I read the
inference code and ran ablations on my own CPU. Everything below is my measurement
unless it's attributed to Mistral.

## Two things the defaults do

**1. Every forecast runs the model twice.** `forecast()` has `symmetric_averaging=True`
by default. It forecasts your series, then forecasts its *negation* (`-context`, with
covariates negated too). It flips that second forecast back (the 1% quantile of the
mirror becomes the 99% quantile of the original, and so on) and averages the two. The
module docstring calls this "clampsym" test-time augmentation. It imposes a symmetry
prior, f(−x) = −f(x), that the network only approximately learned.

**2. It reads up to 32,768 steps of history.** Attention cost grows with the number of
patches, and patches are 32 steps each, so 32k steps means 1,024 tokens per pass.

Here is batch-1 latency on a 4-vCPU Xeon (2.1 GHz), forecasting 48 steps:

| model | ctx 2048 | ctx 2048 + mirror | ctx 8192 | ctx 32768 | ctx 32768 + mirror (default, long series) |
|---|---|---|---|---|---|
| 5M | 11 ms | 22 ms | 36 ms | 121 ms | 212 ms |
| 28M | 28 ms | 62 ms | 64 ms | 251 ms | 614 ms |
| 123M | 90 ms | 174 ms | 217 ms | 873 ms | 1,648 ms |
| 209M | 139 ms | 260 ms | 348 ms | 1,408 ms | **2,940 ms** |

Between the cheapest and the default setting, 209M differs by **21×**. Mistral's quoted
140 ms for 209M is for three weeks of 15-minute data (2,016 steps), and their
8-thread figure is consistent with my 4-thread 260 ms. If you hand it a long history, which is what
the defaults invite, you pay 10–20× that.

So the question is whether that compute shows up in accuracy.

## Setup

- **Data:** 9 datasets from the [Chronos benchmark collection](https://huggingface.co/datasets/autogluon/chronos_datasets):
  15-min electricity, hourly road traffic, ERCOT load, pedestrian counts, KDD-2018 air
  quality, hourly solar, M5 retail (daily, intermittent), exchange rates (daily) and
  30-min NYC taxi. I sampled 32 series per dataset (8 for ERCOT and exchange rate,
  which only have 8). I used 2 forecast origins a week apart, with horizons of 96 / 48 / 28 /
  30 steps depending on frequency.
- **Metric:** weighted quantile loss over the 0.1…0.9 quantiles. I divide it by the
  seasonal-naive baseline on the same windows and take the geometric mean across
  datasets, fev-bench style. **Lower is better**: 0.55 means 45% less loss than repeating
  last week (or last day).
- **Uncertainty:** 95% intervals come from 1,000 bootstrap resamples of series
  within each dataset.
- **Trick:** I call the model's two raw passes myself and combine them exactly as
  `forecast()` does. I checked that my reconstruction matches the library's default
  output bit for bit (max abs diff 0.0). That way every config gets "with mirror" and
  "without mirror" from the same runs.

## Finding 1: the mirror pass does nothing measurable

Here is the loss ratio of default (with mirror) to single pass, at full context:

| model | origin A | origin B |
|---|---|---|
| 5M | 1.004 [0.998, 1.012] | 0.994 [0.987, 1.001] |
| 28M | 0.997 [0.989, 1.003] | — |
| 209M | 0.998 [0.979, 1.017] | 1.000 [0.995, 1.007] |

Every interval contains 1.0, and every point estimate is within 0.6%. You can pass
`symmetric_averaging=False` and halve your latency.

That isn't because the two passes agree. Per series, the forward and mirrored
medians differ by 0.05–0.35 seasonal-MAE units, and per dataset the mirror alone
beats the forward pass by up to 18% (ERCOT) or loses by up to 19% (pedestrians, traffic). The reason averaging doesn't
help is a one-line identity. For squared error, with forward forecast *a* and mirrored
forecast *b*:

```
MSE((a+b)/2) = (MSE(a) + MSE(b)) / 2  −  mean((a−b)²) / 4
                 └── average of the two ──┘    └─ ensemble gain ─┘
```

The gain is a quarter of the squared disagreement. The disagreement is small next to
the errors, because the two passes make almost the same mistakes:

| dataset (209M, full ctx) | MSE fwd | MSE mirror | MSE avg | gain (a−b)²/4 | error correlation |
|---|---|---|---|---|---|
| electricity 15min | 5.456 | 5.532 | 5.482 | 0.012 | 0.996 |
| traffic | 0.518 | 0.552 | 0.527 | 0.008 | 0.966 |
| pedestrians | 0.319 | 0.573 | 0.419 | 0.027 | 0.911 |
| solar | 3.045 | 2.913 | 2.906 | 0.073 | 0.949 |
| M5 | 1.888 | 1.905 | 1.895 | 0.001 | 0.998 |
| exchange rate | 23.61 | 21.10 | 22.32 | 0.037 | 0.997 |

(MSE of the median, in units of each series' seasonal-naive MAE.)

With error correlations of 0.91–0.998, the "ensemble" is really one model asked twice.
What's left is the *prior*. It helps where the data actually is roughly sign-symmetric
(exchange rates, a random walk, improve 2–4%). It hurts where it isn't. Pedestrian
counts have a hard floor at zero, and negating them gives a series with a hard
*ceiling* at zero, a shape real data rarely has. There the 209M mirror pass has 80%
higher error and drags the average down. Across a mixed benchmark the two effects cancel.
On your own data the effect is predictable: off for counts and demand, maybe on
for price-like series.

## Finding 2: more context isn't better, and you can't recover it

Here is the context sweep (single pass, origin A):

| model | 512 | 2048 | 8192 | 32768 |
|---|---|---|---|---|
| 5M | 0.633 | **0.579** | 0.588 | 0.600 |
| 28M | 0.610 | **0.578** | 0.593 | 0.599 |
| 209M | 0.595 | 0.546 | **0.545** | 0.562 |

512 steps is clearly too short. Beyond about 2k, the curve is flat or gets worse. The
paired comparisons (2048 single pass vs. the full default) are:

| model | origin A | origin B |
|---|---|---|
| 5M | 0.960 [0.943, 0.975] | 0.974 [0.957, 0.999] |
| 28M | 0.968 [0.952, 0.983] | — |
| 209M | 0.974 [0.955, 0.990] | 1.017 [0.986, 1.052] |

For the small models, 2k context with no mirror is **better** at both origins, and 10–20×
faster. For 209M it's a wash: 2.6% better at one origin, 1.7% worse at the other.

I expected a specific mechanism and it turned out wrong, which is worth reporting. Timestral
normalizes each patch with the *cumulative* mean and standard deviation of everything
before it, and squashes with `asinh`. Forecasts come out in the frame of the last
context patch, so at 32k steps that frame is set by years of hourly history. If the
recent level has drifted from that, the recent dynamics land in the compressed tail of
`asinh`. It's a neat story, and the 5M data partly fits it: the loss from long context
correlates with how far the last 2k steps drift from the full-history mean (Spearman
0.28, p < 0.001).

So I tested it directly. I monkeypatched the scaler so the full 32k context is
normalized with **trailing 2,048-step** statistics, both in the encoder and in the
horizon frames:

| | ctx 2048 | ctx 32768 | ctx 32768, windowed normalization |
|---|---|---|---|
| 5M | 0.579 | 0.600 | 0.607 |
| 209M | 0.546 | 0.562 | 0.560 |

It recovers nothing. For 209M the drift correlation is −0.06. So the normalization
frame isn't the cause. The extra degradation comes from what attention does with the
old patches, not from how they're scaled.

## Finding 3: 5M is most of the way there

At the same 2k single-pass setting, 28M is indistinguishable from 5M (ratio 0.999
[0.983, 1.013]). 209M is 5.6% better at origin A and 7.1% better at origin B. Against
the full 209M default, 5M at 2k is 3–10% behind and about **270× cheaper** (11 ms vs
2,940 ms).

## What I'd actually run

```python
f = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")   # or -5M
fc = f.forecast(history[-2048:], prediction_length=48,
                symmetric_averaging=False)      # keep clip_nonnegative=True
```

That's about 21× faster than the defaults on long histories, within a few percent on accuracy
in either direction for 209M, and slightly better for 5M and 28M. If you need the last
few percent, sweep context on your own data. The best length differed by model and
by origin here, and no single value won everywhere.

## Caveats

- This is univariate only, with no covariates. Timestral's covariate attention is a big
  part of its pitch, and none of this tests it.
- These datasets are standard benchmark material, and Mistral post-trained on real
  data, so absolute skill may be flattered by overlap. The ablations are relative
  within one model, which makes them less exposed to that.
- 32 series per dataset and two origins. The intervals reflect series sampling, not
  every source of variance.
- Latencies are from one 4-vCPU VM in fp32 PyTorch. A GPU changes the absolute numbers,
  but the mirror is still a second full forward pass.

*Sources: [Mistral's announcement](https://mistral.ai/news/timestral),
[mistralai/timestral](https://github.com/mistralai/timestral) (inference code, read at
HEAD on 2026-10-20), [Chronos datasets](https://huggingface.co/datasets/autogluon/chronos_datasets).
Benchmarks, analysis and conclusions are my own.*
