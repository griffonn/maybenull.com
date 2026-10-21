---
title: "DeepInfer's variable draft lengths, simulated: what a confidence head buys"
description: DeepSeek's inference engine decides how many speculative tokens to verify per request by ranking a learned confidence score. I read the scheduler, ported it to Python, and simulated where it beats a fixed draft length, by how much, and what kind of confidence error actually hurts.
date: 2026-10-21
draft: false
---

Speculative decoding has a knob that most engines set once and forget: how many
draft tokens to verify per step. Verify too few and you leave accepted tokens on
the table. Verify too many and, once the batch is big enough to be
compute-bound, you pay for rejected tokens in every forward pass.

[DeepInfer](https://github.com/deepseek-ai/DeepInfer), the engine DeepSeek
open-sourced this week (I [looked at its SWA replay trick](/posts/swa-bounded-replay/)
yesterday), ships a different answer behind `DEEPINFER_ENABLE_MTP_VARLEN=1`. A
small learned head predicts how likely each draft token is to be accepted. The
scheduler then decides, per step, both the total number of tokens to run and
how to split them across requests. The README gives it a sentence. The
mechanism is in about 150 lines of C++ and a TileLang kernel. Nobody has
published numbers for what it buys over a fixed draft length.

I can't run V4.1-Flash on SM100s, so this post does two things instead. First,
it explains the scheduler from the source (everything cited below is at commit
[`774ce24`](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)).
Second, it ports the scheduler to Python and simulates it against the baselines
it replaces, on a synthetic step-time curve. The code reading reflects what the
repo does. The numbers come from my simulation and depend on its assumptions,
which I spell out.

## What the code does

**1. A confidence head per draft position.** The drafter (a block "DFlash"
drafter, up to `DEEPINFER_USE_NEXTN_DECODE=5` tokens) has an extra linear probe.
It is `sigmoid(cat(hidden, markov_emb) @ w / T)`
([`tilelang_ops/dflash.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/dflash.py)),
where `markov_emb` is a low-rank embedding of the previous draft token. The
same embedding also adds a bigram-style correction to the draft logits. The
temperature `T` can be set per position
(`DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` accepts a comma-separated list).
The sampler logs the summed `|accept_prob − confidence|` over verified
positions as `avg_confidence_error`.

**2. Cumulative products.** Draft *j* only gets accepted if drafts 1…*j−1*
were, so the C++ side turns per-position confidences *c₁…c₅* into
`accept_rates[j] = c₁·c₂·…·cⱼ`
([`scheduler.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp),
`process_MTPInfo`). If the confidences are calibrated, that number is the
probability that draft *j* produces a token, so summing it over the drafts you
verify gives the expected accepted count.

**3. One global sort, then a scan over budgets.** `MTPVarlen::schedule`
([`mtp_varlen.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp))
puts every (request, position) pair from the whole batch into one list and
sorts it by accept rate, descending. It then walks the list, adding one token
at a time, and scores each prefix as *expected tokens ÷ predicted step time*.
Step time comes from a decode curve profiled at startup for each CUDA graph
size, plus a fitted context-length slope and a median sampler time. The best
prefix sets `total_q`, and each request verifies as many drafts as it won
places in that prefix. Here is my Python port. Its allocation step matched a
numpy port of the repo's reference implementation of the GPU kernel on 2,000
random batches:

```python
def varlen_schedule(perf, rates_bsz, rates_alloc):  # rates: [N, 5] cumprods
    n = len(rates_bsz)
    q_bsz = np.sort(rates_bsz.ravel())[::-1]
    order = np.lexsort((np.repeat(np.arange(n), 5), -rates_alloc.ravel()))
    best_tp, best_bsz, expected, bsz = 0.0, n, float(n), n
    for k in range(len(q_bsz) + 1):
        if k: expected += q_bsz[k - 1]; bsz += 1
        if bsz > perf.sizes[-1]: break
        _, t = perf.bucket(bsz)          # time of the smallest graph >= bsz
        if expected / t > best_tp: best_tp, best_bsz = expected / t, bsz
    drafts = np.zeros(n, int)
    for idx in order[: best_bsz - n]: drafts[idx // 5] += 1
    return drafts
```

**4. Two different rounds of rates.** Scheduling runs asynchronously ahead of
the GPU, so the budget is chosen from round *N−2* confidences and the split
from round *N−1*. With zero-overhead scheduling enabled, the split is then
recomputed on the GPU (`compute_seqlens_q_by_confidence_rates`) once the
previous step's draft confidences exist.

## Three things that follow from the code alone

These are my observations, not claims from the repo.

**It is exactly optimal for its own beliefs.** Each row of cumulative products
is non-increasing, since multiplying by a number ≤ 1 can't increase a float. A
global top-*B* by value therefore always picks a prefix of each row. The
tie-break (lower sequence index first) keeps that true
when values are equal. For a fixed budget *B*, maximizing expected accepted
tokens is a unit-cost knapsack, and taking the top *B* items solves it exactly.
Scanning every *B* then maximizes expected throughput. So with calibrated
confidences and an accurate step-time curve, no other per-step allocation can
do better in expectation. Every loss comes from errors in the beliefs.

**Padding turns into speculation.** CUDA graphs come in fixed sizes (1, 2, 4,
then every 8 by default,
[`build_graph_batch_sizes`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/cuda_graph_runner.py)).
Inside one size the step time is constant while expected tokens keep rising, so
the scan never stops partway through a graph. In 1,200 simulated steps,
`total_q` landed exactly on a graph size every time. Any slot a fixed-*k*
engine would pad gets a low-confidence draft here.

**Without the head, it means "draft everything".** When a request has no
confidence values, the scheduler pads them with 1.0. Every extra token then
looks like a guaranteed token, the scan runs to the largest graph that fits,
and you get fixed *k* = 5. In my runs at mid-to-high load that fallback was
20–50% slower than the best fixed *k*. The flag only helps with a model whose
checkpoint includes the confidence head.

## The simulation

**Step time.** I used a roofline, `T(g) = max(10 ms, g · t_tok) + 1 ms`, with
the knee at *g* = 512 tokens: memory-bound below it, compute-bound above. It
is evaluated only at DeepInfer's graph sizes. I also tried a smooth version,
`(10⁴ + (g·t_tok)⁴)^¼`. Neither is a measured DeepSeek curve, so read the
results against *load* (batch size ÷ knee) rather than as milliseconds.

**Acceptance.** Each request has a base acceptance rate *aᵢ* ~ Beta(7, 3)
(mean 0.7, standing in for code vs. chat vs. creative prompts). Each draft
position's conditional acceptance probability is
`sigmoid(logit(aᵢ) + ε)` with ε ~ N(0, 1), redrawn every step (some tokens are
easy, some aren't). A step's outcome is the leading run of successes among the
verified drafts, plus the always-produced token.

**Baselines.** I gave the baselines every advantage. *Best fixed k*: the
same draft count for every request, with *k* ∈ {0…5} chosen in hindsight for
each batch size. *Best threshold*: verify draft *j* while its cumulative
confidence is ≥ θ, with θ chosen in hindsight (roughly the dynamic-length
policies in EAGLE-2 and SpecDec++, applied per request with no global budget).
Every policy sees the same random draws. Each cell is 1,500–2,000 decode steps.

### Gain over the best fixed k, by load

| batch N | load N/512 | best fixed k | varlen gain (roofline) | varlen gain (smooth) |
|---:|---:|---:|---:|---:|
| 64  | 0.12 | 5 | +0.0% | +1.6% |
| 110 | 0.21 | 4 | +10.7% | +8.3% |
| 140 | 0.27 | 3 | +13.3% | +9.2% |
| 200 | 0.39 | 2 | +15.3% | +9.8% |
| 280 | 0.55 | 1 | +9.5% | +9.4% |
| 400 | 0.78 | 1 | +13.4% | +8.3% |
| 480 | 0.94 | 0 | +6.7% | +9.4% |
| 560 | 1.09 | 0 | +0.4% | +8.5% |

```
gain vs best fixed k (roofline step time)
load 0.12 |                     0.0%
load 0.21 |##########          10.7%
load 0.27 |#############       13.3%
load 0.39 |###############     15.3%
load 0.55 |#########            9.5%
load 0.78 |#############       13.4%
load 0.94 |######               6.7%
load 1.09 |                     0.4%
```

The pattern is the main result. At small batches every extra token is free,
*k* = 5 is optimal, and there is nothing to optimize. Deep in the compute-bound
regime the roofline says speculation never pays, and varlen correctly turns it
off. The gains are in between, where most production decode batches sit. There
the best fixed *k* steps down 5 → 4 → 3 → 2 → 1 → 0, and varlen does about
10–15% better. With the smooth curve, which matches real hardware better
because it has no sharp knee, the band is wider and flatter: 8–10% from load
0.2 through past 1.0.

### Where the gain comes from

At N = 200 (load 0.39) I turned the two sources of variation on one at a time:

| acceptance model | vs best fixed k | vs best threshold |
|---|---:|---:|
| A. every request 0.7, no per-token variation | +4.2% | +4.2% |
| B. every request 0.7, per-token variation | +11.0% | +4.2% |
| C. requests vary, no per-token variation | +10.4% | +1.1% |
| D. both (the setting above) | +14.6% | +3.4% |

Row A is the surprise. With no information at all (every confidence equal),
varlen still wins 4%. It verifies 1.56 drafts per request on average where a
fixed policy has to choose 1 or 2. That lands `total_q` at exactly 512, the
knee, instead of 400 or 600. The tie-break
decides who gets the extra token: lowest sequence index. In a real engine the
confidences are continuous, so exact ties should be rare.

The *vs threshold* column is the honest comparison for anyone already running
dynamic draft lengths. A per-request threshold, tuned in hindsight, recovers
most of the gain, and the global budget adds another 1–4% on top. That last bit
comes from setting the cutoff each step from the batch's composition and the
graph sizes, rather than one θ for all time.

### What kind of confidence error hurts

DeepInfer reports one diagnostic, the mean `|accept_prob − confidence|`. I
degraded the oracle head two ways at N = 200 and measured that same metric
alongside the AUC of confidence against actual acceptance:

| confidence head | mean \|p − c\| | AUC | gain vs fixed k |
|---|---:|---:|---:|
| oracle | 0.000 | 0.780 | +14.6% |
| under-confident (logit × 0.5) | 0.099 | 0.780 | +12.4% |
| over-confident (logit × 2) | 0.102 | 0.780 | +13.7% |
| logit noise σ = 0.5 | 0.067 | 0.759 | +13.4% |
| logit noise σ = 1 | 0.129 | 0.714 | +10.9% |
| logit noise σ = 2 | 0.225 | 0.643 | +6.1% |
| per-request average only (no per-token signal) | 0.142 | 0.660 | +9.6% |

Miscalibration that keeps the ordering is nearly free. Doubling every logit
gives a head with *higher* logged error than σ = 0.5 noise, yet it schedules
better. Noise, which scrambles the ordering, is what costs throughput. This
makes sense given the design: the split depends only on how the cumulative
products rank, and calibration only enters through the budget choice, where
errors average out over hundreds of requests. (Using round *N−2* rates for the
budget made no measurable difference in any run, for the same reason.)

Two practical takeaways. First, `avg_confidence_error` is a weak health
signal for this scheduler. If you train or fine-tune the head, track a ranking
metric too. Second, the per-request-average row is the cheap alternative, an
exponential moving average of each request's acceptance rate with no learned
head. It captures about two-thirds of the gain. The per-token head is worth
the last third, provided its AUC stays well above that baseline.

## Caveats

- The step-time curves are synthetic. The knee position and how sharp it is
  determine *where* the gain band sits. Whether there is a band at all depends
  mainly on step time rising with tokens past some batch size.
- Acceptance here is independent across positions given the per-token
  difficulty. Real failures cluster (a draft that goes off the rails tends to
  stay off them), and I didn't model that.
- I did not model the drafter's own cost or the sampler time, which DeepInfer
  includes in its estimate. Both push toward fewer drafts.
- Nothing here measures DeepSeek's actual head. The question it answers is how
  good the head needs to be, and which metric shows whether it is.

## Sources

- DeepInfer at commit `774ce24`:
  [`mtp_varlen.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp),
  [`scheduler.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/scheduler.hpp),
  [`schedule_mtp_varlen.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py),
  [`dflash.py` (confidence head)](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/dflash.py),
  [`cuda_graph_runner.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/cuda_graph_runner.py)
- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)
- Simulation: my own (numpy). The scheduler port is above. The rest is the
  acceptance and step-time model described in the post.
