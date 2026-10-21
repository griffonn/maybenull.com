---
title: "SlopCodeBench's strict pass: how much of the leaderboard is noise?"
description: "Strict passes are sums of clean prefixes, so a 3-checkpoint gap means little. I modeled the noise and checked the model against the published spread."
date: 2026-10-21
draft: false
---

HumanLayer's October [SlopCodeBench update](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/benchmarking-new-models-on-slop-code-bench.md)
reached the Hacker News front page as "xhigh considered harmful". The source
data is ten agent campaigns over the full v1.0 catalog (36 problems, 196
checkpoints), each run once. Strict passes ranged from 24 to 32. In the three
Medium-vs-XHigh pairs, strict passes moved by −3 (Opus 5.5), −4 (GPT-6 SOL)
and +3 (GPT-6 Astra), while XHigh cost 2.1–4.2× as much.

The author is careful here. They say plainly that one campaign per setting
can't separate effort from run-to-run variation. I wanted to know how big
that variation is. Could a three-checkpoint gap be noise? How many runs would
it take to tell? The answer turned out to depend on a structural property of
the strict metric that's easy to miss.

## Strict passes are a sum of 36 clean prefixes, not 196 trials

SlopCodeBench reveals a problem one checkpoint at a time. At checkpoint *k*
the harness runs that checkpoint's tests plus every earlier checkpoint's tests
as a regression suite. A strict pass requires all of them to be green, so the
first defect in a problem rules out strict passes for every later checkpoint
of that problem. Later work could in principle fix it, but the author reports
that this didn't happen in practice.

So for each problem the strict count is the length of its *clean prefix*:

```
problem with 6 checkpoints:   ✓ ✓ ✗ · · ·   → contributes 2
problem with 8 checkpoints:   ✗ · · · · · · · → contributes 0

strict(run) = Σ over 36 problems of clean_prefix_length(problem)
```

The suites behind these prefixes are large. I cloned the
[problem repository](https://github.com/gabeorlanski/scb-problems) and ran
`pytest --collect-only` for every checkpoint. Collection worked cleanly for
25 problems. My count for `circuit_eval` is 566 cumulative tests, which
matches the number the HumanLayer post gives for the same suite. The median
problem reaches 216 tests by its final checkpoint, and `test_translator`
reaches 2,069. One failing case anywhere in that pile ends the prefix.

This structure matters for noise. A sum of 196 independent pass/fail
outcomes would be fairly stable. A sum of 36 truncated geometric variables
isn't: one early stumble on an 8-checkpoint problem can wipe out several
passes at once.

## Model 1: every checkpoint is a fresh coin flip

I started with the simplest model. Each problem *p* gets a per-checkpoint
clean probability *q_p*, drawn from a Beta distribution and fixed for the
benchmark. Each run then flips a coin with probability *q_p* at each
checkpoint until the first failure. The run-to-run variance has a closed
form: if *r_j* is the probability of being clean through checkpoint *j*,
then *E[L] = Σ r_j* and *E[L²] = Σ (2j−1)·r_j*.

I fit this to two published facts. The ten campaigns average **27.7**
strict passes, and they average **1.4** strict *final* checkpoints out of 36
(Table 3 of the update). Across the plausible range of heterogeneity, the
model predicts a single-run standard deviation of **5–6.7 checkpoints**.

That can't be right, and the published table shows why. The spread across
ten *different* campaigns, covering four model families and two effort
levels, has a standard deviation of only **2.45**. True differences between
models can only add to the spread that noise produces. If one run's noise
were σ = 6.5, the chance that ten campaigns would land this tightly is about
0.15% (χ² with 9 degrees of freedom). Even σ = 4 only gets to about 5%.

So the coin-flip picture is wrong. Most checkpoint outcomes must be shared:
every agent passes the same easy checkpoints and fails at the same walls.

## Model 2: shared difficulty, calibrated to the spread

Next I gave every *checkpoint* its own pass probability π. The values come
from a two-level Beta model: problem-level difficulty, then per-checkpoint
scatter around it. They're fixed for the benchmark and shared by every run,
and a concentration parameter *c* controls how close the π values sit to 0
or 1. I fit the mean to 27.7, the problem-level spread to 1.4 strict finals,
and then swept *c*:

```
problem κ=4, mean fit to E[strict]=27.7          (300 benchmark draws per row)

     c   run SD   E[finals]   "swing" ckpts   "sure" ckpts   share from ck1   ck1+ck2
  1000    6.48      1.54          72.4            0.3            0.49           0.73
     4    6.12      1.73          63.6            2.6            0.48           0.71
     1    4.91      1.54          42.2            8.3            0.49           0.72
   0.4    3.76      1.66          25.3           15.6            0.49           0.72
   0.2    2.97      1.77          16.2           21.2            0.48           0.72
   0.1    2.19      1.65           8.9           23.7            0.49           0.73
```

A "swing" checkpoint is one whose strict-pass probability is between 0.1 and
0.9. A "sure" checkpoint is ≥ 0.9.

For the noise alone to stay under the observed 2.45 spread, the benchmark
needs *c* ≈ 0.1–0.15. In that regime about 23 checkpoints are near-certain
strict passes for everyone. Only about **10 of the 196 checkpoints are
actually in play**. This is an upper bound: if the models really do differ,
some of the spread is signal and there are even fewer swing checkpoints.
Other heterogeneity settings that also reproduce ~1.4 finals give the same
picture.

The last two columns show something the headline rate hides. In every
calibration that fits the finals count, about **half of all strict passes
are checkpoint 1**, and about 70% are checkpoints 1–2. Checkpoint 1 means
writing a program from a fresh spec, with no codebase to extend yet. That's
the part of SlopCodeBench least about long-horizon code evolution, which is
what the benchmark exists to measure. This share comes from the model, not
from published per-checkpoint data. But it agrees with the earlier
HumanLayer run on three problems, where 3 of the winner's 4 strict passes
were the opening checkpoints of a single problem, and the only pass shared
by all three models was a checkpoint 1.

## What this means for "xhigh considered harmful"

Assume the worst-case noise the data allows, σ ≈ 2.45 per campaign. The
difference between two campaigns then has σ ≈ 3.5:

```
pair          Δ strict   z (σ=2.45)   z (σ=1.5)
Opus 5.5         −3        −0.87        −1.41
GPT-6 SOL        −4        −1.15        −1.89
GPT-6 Astra      +3        +0.87        +1.41
pooled mean    −1.33       −0.67        −1.09
```

None of these comes close to conventional significance, alone or pooled,
even if the true noise is well below the bound. The full leaderboard fits a
null result just as well. The expected range of ten draws from one
distribution is about 3.08σ, or about 7.5 checkpoints at σ = 2.45. The
observed range is 8 (32 − 24). Ten copies of a single agent would produce a
leaderboard like this about 38% of the time. To be fair, that's the bound
case. If real noise is smaller, the gaps could be real. The data can't tell
you which.

To detect a true 3-checkpoint effect (α = 0.05 two-sided, 80% power), you
need this many runs per setting:

```
noise σ    runs per arm    Opus 5.5 XHigh cost at $817/run
  1.0            2                 $1.6k
  1.5            4                 $3.3k
  2.0            7                 $5.7k
  2.45          11                 $9.0k
```

So the most expensive cell in that table would need somewhere between
$1.6k and $9k of repeats before "medium beats xhigh" is a finding rather
than a coin toss.

## A metric with more signal per dollar

The structure also suggests what to read instead. Core pass doesn't absorb
failures: a model can fail checkpoint 2 and still get Core credit at
checkpoint 5. Every checkpoint therefore contributes its own information,
with no clean-prefix truncation amplifying one early mistake. Its spread
across campaigns is more than twice as wide (σ = 5.6 vs 2.45, 129–145 of
196). Whether that spread is signal still takes repeats to establish: a
binomial-style ceiling on per-checkpoint Core noise is about 6.4. But at least
it isn't decided by a handful of swing checkpoints. The per-test failure
counts in Table 3, which run in the thousands, carry far more bits than 0–3
strict finals.

None of this argues against strict pass as the headline. "Did you break
anything?" is the right question for long-horizon coding. But it's a
low-resolution instrument. With one run per cell, it ranks about ten swing
checkpoints, not 196.

## Method notes and caveats

- **My analysis:** the clean-prefix decomposition, both noise models, the
  calibration and the power numbers. The scripts are about 60 lines of
  NumPy/SciPy. The core of the variance computation is below.
- **From the sources:** all campaign results, costs and the strict/Core
  definitions come from the
  [HumanLayer update](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/benchmarking-new-models-on-slop-code-bench.md)
  and [earlier runs](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/benchmarking-opus-5-on-slop-code-bench.md).
  Checkpoint counts and test collection come from
  [scb-problems](https://github.com/gabeorlanski/scb-problems). The harness
  is [SprocketLab/slop-code-bench](https://github.com/SprocketLab/slop-code-bench),
  and the paper is [arXiv:2603.24755](https://arxiv.org/abs/2603.24755).
- **Assumption:** failures absorb. This is the author's observation, not a
  guarantee. Occasional later fixes would make strict slightly *less* noisy
  than modeled.
- **Assumption:** the spread bound treats all ten campaigns as exchangeable
  under the null. Using them to bound noise is conservative in one direction
  only: it can overstate the noise, never understate it.
- **Not done:** I didn't rerun any agents. At these prices the right
  experiment is three repeats of one cheap cell, such as GPT-6.1 SOL Medium
  at about $179 per run. That would pin σ directly, and I'd love to see it.

```python
# reach[j] = P(clean through checkpoint j) for one problem, given fixed pass probs pi
reach = np.cumprod(pi)
j = np.arange(1, len(reach) + 1)
E_L  = reach.sum()                  # expected clean-prefix length
E_L2 = ((2 * j - 1) * reach).sum()  # since P(L >= j) = reach[j]
var  = E_L2 - E_L**2                # per-problem run variance; sum over problems
```
