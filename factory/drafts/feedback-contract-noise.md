---
title: "One standard error lets one noise submission in six through"
description: A new paper protects an ML agent's validation set by releasing a score only when a submission beats the best by one paired standard error. That test passes about 1 in 6 pure-noise submissions, so against small edits it slows fake gains (4.3 vs 5.8 points after 1,000 tries) rather than stopping them. Against the classic attack it holds, and it keeps most real gains.
date: 2026-10-22
draft: false
---

An ML agent that reads its validation score after every attempt will, given
enough attempts, improve the validation score and nothing else. A new paper
fixes this by showing the agent a new score only when a submission beats the
best one by one standard error. I fed that rule 1,000 submissions of pure noise.
It reported 4.3 points of accuracy that don't exist. Exact feedback reported 5.8.

## Context

[An Explicit Feedback Contract for Iterative ML-Agent Harnesses](https://arxiv.org/abs/2610.28798)
(Yao et al., Walmart Global Tech, October 2026) looks at agents that develop
models in a loop: edit code, train, read a validation score, decide what to try
next. Each score is an honest held-out estimate. Keeping whichever edit scored
best is not. After enough rounds the champion is partly a model that happens
to fit *this* validation split. That is adaptive overfitting.

The paper's fix is a harness "contract": the agent can't read validation labels
or predictions, the test set is touched once at the end, and the score is
released only when a submission beats the champion clearly enough. "Clearly
enough" is the interesting part. The harness computes each example's loss for
the candidate and for the champion, takes the standard deviation *s* of the
per-example differences, and accepts only if the candidate's mean loss is lower
by more than *s*/√*n*. That is one standard error of a paired comparison.
Otherwise the old score is repeated.

The rule is the "parameter-free Ladder" from Blum and Hardt's
[2015 leaderboard paper](https://arxiv.org/abs/1502.04585), down to the
all-zeros starting loss vector and the rounding to 1/*n*. Blum and Hardt call
the test "primarily heuristic" and put it at roughly a 0.15 significance level.

On a control task with permuted labels, the paper reports a spurious gain of
0.0157 under exact feedback and 0.0033 under the contract. Exact feedback ran
all 30 allowed submissions; the contract stopped after about 10.

## The question

One standard error is a one-sided z-test at z = 1. If the candidate is no better
than the champion, it still passes about 16% of the time. One false accept is
cheap. The question is whether false accepts stack: a searcher that keeps
nudging its current best might climb on noise one 16% step at a time.

Blum and Hardt tested the rule against an attack that submits *independent*
random guesses. Each accepted guess raises the bar, so later guesses rarely
clear it. Agents don't submit independent guesses. They submit the champion
plus a small change. The paper's experiments run 30 LLM-agent submissions, which
is too few to see this, and they don't say how much successive submissions
differ. So I checked three things:

1. How much fake gain each release rule lets through, as a function of how many
   predictions each submission changes.
2. How many predictions realistic edits actually change.
3. What each rule costs on a task where there is something real to learn.

## Setup

Everything is numpy and scikit-learn on a 4-core CPU, scored with 0-1 loss
(error rate). For 0-1 loss the per-example differences are well defined, and
rounding to 1/*n* changes nothing. Validation sets have *n* = 25,000 rows, the
size of the paper's Criteo validation split. Three release rules:

| rule | accepts a submission when its validation error is below the champion's by more than |
|---|---|
| exact | 0 (any improvement) |
| paired SE (the paper) | std(paired loss differences) / √*n* |
| fixed margin | 0.005 (one of the margins in the paper's Table 4) |

**Pure noise.** Labels are coin flips, so every model's true accuracy is 50% and
any reported gain is selection. The searcher flips *m* random predictions of the
current champion and submits the result, 1,000 times, 5 seeds:

```python
cand = champ.copy()
cand[rng.choice(N, m, replace=False)] ^= 1
Lc = (cand != y).astype(float)                    # per-example 0-1 loss
if Lc.mean() < R - np.std(Lc - L) / np.sqrt(N):   # the paper's rule
    champ, L, R = cand, Lc, Lc.mean()
```

I also ran Blum and Hardt's boosting attack: submit 1,000 random label vectors,
then submit the majority vote of the ones the harness called good.

**A scripted hill-climber.** The paper's agents are Codex and Claude. Mine is a
script, so it is cruder but reproducible. It uses a HistGradientBoostingClassifier
on synthetic tabular data (`make_classification`: 30 features, 10 informative,
30% label noise; 20k train, 25k validation, 50k test). The `null` variant
replaces the labels with coin flips. Two searchers, 200 submissions each, 5 seeds
per rule:

- **tweak:** change one hyperparameter of the champion by one step (learning
  rate ×1.25, ±4 leaves, a new random seed, and so on).
- **ensemble:** add one new model (one tweak, fresh seed) to the champion's
  probability-averaged ensemble. This is "add it to the blend if validation
  improves", which is common practice. It is also a polite version of the
  boosting attack.

I report gains from the first submitted model. **Inflation** is validation gain
minus fresh-test gain, the part of the reported progress that isn't real.

## Results

### Pure noise

![Reported accuracy gain after 1,000 pure-noise submissions vs. predictions changed per submission, for three release rules](/lab/feedback-contract-noise/noise-sweep.png)

| predictions changed per submission (m) | exact | paired SE | fixed 0.005 |
|---:|---:|---:|---:|
| 2 | 1.9 | 1.9 | 0.0 |
| 10 | 4.2 | 2.8 | 0.0 |
| 100 | 5.8 | 4.3 | 0.0 |
| 1,000 | 3.4 | 2.7 | 0.3 |
| 10,000 | 1.2 | 0.9 | 0.8 |
| boosting attack | 4.5 | 0.7 | 0.7 |

*Reported validation gain in accuracy points after 1,000 submissions, mean of 5
seeds (spread about ±0.2–0.3). Fresh-test gain is within ±0.3 of zero in every
cell.*

The paired-SE rule stops the boosting attack, as Blum and Hardt showed: 0.7
points, against 4.5 for exact feedback. Against small edits it only slows things
down. At *m* = 100 it lets through 74% of exact feedback's fake gain. Real
accuracy on fresh labels is 50% throughout.

The first-step pass rate matches the z = 1 arithmetic. With a champion that
hasn't been selected yet, the rule accepted 16–18% of noise candidates for *m*
from 10 to 10,000. At *m* = 2 it accepted 24%, because the loss differences are
coarse at that size.

There is also an edge case in how the formula is written. With the population
standard deviation (`np.std`'s default), a one-row improvement always clears the
bar: the threshold is √(1 − 1/*n*)/*n*, just under the 1/*n* the score moves by.
At *m* = 1 the rule then behaves exactly like exact feedback. Blum and Hardt
specify the sample standard deviation, which blocks *m* = 1. It doesn't help at
*m* = 2.

The curves rise and then fall because the gain eats itself. A champion that is
lucky on the validation set loses some of that luck with every flipped row: of
the *m* rows it changes, more were right than wrong. With small *m* each
accepted step is tiny. With large *m* each candidate gives back most of the luck
it has collected. The worst case sits in between, around 50–100 rows per edit
at *n* = 25,000. The fixed margin blocks everything below about 1,000 rows,
because a 0.005 jump is 125 rows at once.

### A scripted hill-climber

Realistic edits are not small in this sense. On the coin-flip labels, a one-step
tweak changes a third of the validation predictions (median 8,300 of 25,000). A
boosted tree fit to noise reshuffles its noise whenever you touch it. On
learnable labels a tweak changes about 600 predictions, and adding a member to
a growing ensemble changes 36–107 (median, depending on how big the ensemble has
grown).

| searcher, labels | rule | accepted (of 199) | val gain | test gain | inflation | inflation, patience 5 |
|---|---|---:|---:|---:|---:|---:|
| tweak, coin flips | exact | 3.6 | +0.46 | −0.07 | +0.52 ± 0.50 | +0.25 (stops at 7.6) |
| | paired SE | 0.4 | +0.18 | −0.05 | +0.23 ± 0.34 | +0.15 (6.2) |
| ensemble, coin flips | exact | 8.8 | +0.64 | −0.15 | +0.79 ± 0.18 | +0.35 (13.0) |
| | paired SE | 1.8 | +0.56 | +0.08 | +0.48 ± 0.40 | +0.09 (6.4) |
| | fixed 0.005 | 0.6 | +0.33 | −0.08 | +0.41 ± 0.39 | +0.18 (6.8) |
| ensemble, learnable | exact | 17.0 | +0.54 | **+0.38** | +0.16 ± 0.09 | +0.08 (19.8) |
| | paired SE | 4.2 | +0.46 | **+0.32** | +0.14 ± 0.09 | +0.07 (10.4) |
| | fixed 0.005 | 0.0 | 0.00 | **0.00** | 0.00 | 0.00 (6.0) |

*Accuracy points, mean ± SD over 5 seeds, 200 submissions. "Patience 5" replays
the same runs with a stop after 5 consecutive submissions without an update
(the paper's contract has a patience stop; it doesn't state p).*

Three readings. First, a scripted searcher on 25,000 validation rows doesn't
overfit much: under a point of inflation after 200 submissions, even with exact
feedback. The paired-SE rule roughly halves it on coin-flip labels, but the
seed-to-seed spread is as large as the difference.

Second, the patience stop does as much as the threshold. Exact feedback plus
"stop after 5 rounds without improvement" lands within noise of the paired-SE
rule without one. With the same patience of 5, exact feedback stopped after
8–20 submissions and the paired-SE rule after 6–10: a threshold that accepts
less often makes the patience counter fire sooner. The paper's exact-feedback
agents never stopped before the cap, so its LLM agents kept finding
"improvements" more often than my scripts did.

Third, the cost. On learnable labels, ensembling is worth +0.38 points on fresh
test data. The paired-SE rule keeps +0.32 of it with a quarter of the updates.
The fixed 0.005 margin keeps nothing, because no single real improvement here
is that big. The margin that Blum and Hardt's theory needs is worse:
*n*<sup>−1/3</sup>·log<sup>1/3</sup>(*kn*) is 0.085 at *n* = 25,000 and
*k* = 200, more than 20 times the whole real gain. The paper puts a floor of
that form, times a constant *c*, under its rule, and doesn't give *c*.

## What it means

The paired-SE rule is a sensible default with a narrow guarantee. Its
protection depends on what your submissions look like. It holds against
independent guesses, because each accepted guess raises the bar for the next
one. It barely slows a searcher that makes many correlated small edits, because
every edit is a fresh 16% chance. Agents that tune, blend and patch are closer
to the second kind. On the evidence here, the thing that actually limits damage
is the number of submissions: a patience stop, a budget, or both.

What I'd take from it if I were writing an agent harness:

- Log *m*, the number of validation predictions each submission changes. It
  tells you which regime you're in, and the agent doesn't need labels to compute
  it.
- Use the paired-SE rule over a fixed margin. A fixed margin is either too big,
  blocking real gains, or too small to matter, and you can't tell which in
  advance.
- Keep the patience stop and treat it as the main defense. Count accepted
  updates, not raw improvements.
- Score the final model once on a test set the loop never saw. That is the only
  number in this post that never lied.

**Limits.** My searchers are scripts, not LLM agents. An LLM agent edits code,
not just hyperparameters, and its edit sizes may be different. Everything here
is 0-1 loss. The paper scores Criteo with AUC, which is not an average of
per-example losses, and it doesn't say what vector the standard deviation is
taken over. Validation size is fixed at 25,000. Smaller sets make every effect
here larger. Five seeds separate the pure-noise curves cleanly; for the
hill-climber they don't separate paired SE from exact feedback on inflation. I
didn't test Thresholdout or other differential-privacy mechanisms.

## Reproduce it

Everything is in
[`/code/feedback-contract-noise/`](/code/feedback-contract-noise/README.md), also as one
[tarball](/code/feedback-contract-noise.tar.gz): the simulator, the hill-climber,
per-submission logs for all 60 runs, and the analysis and plotting scripts.

```sh
pip install -r requirements.txt   # numpy 2.1.3, scikit-learn 1.5.2, matplotlib 3.9.2
python sim.py                     # pure noise + boosting attack, ~1 min
./run_agent.sh tweak              # ~10 min on 4 cores
./run_agent.sh ensemble           # ~25 min on 4 cores
python analyze.py && python plot.py
```
