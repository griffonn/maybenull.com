---
title: "How wrong is DeepSeek's SWA bounded replay? I measured it on Gemma"
description: DeepInfer rebuilds its sliding-window KV cache from the last 128 tokens and admits the result is approximate. I reproduced the trick on Gemma-3-270M and measured the error against exact inference, bf16 noise and throwing the cache away.
date: 2026-10-20
draft: false
---

DeepSeek open-sourced [DeepInfer](https://github.com/deepseek-ai/DeepInfer) this week,
the inference engine behind DeepSeek-V4.1-Flash. Most of what it does is known
engineering: expert parallelism, MTP speculative decoding, batch-invariant
kernels. One item in the
[deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md)
is unusual because it says outright that it changes the model's outputs.

They call it **SWA bounded replay**. Their KV cache has two parts. A "block" KV
cache handles long-range attention and is cached by prefix. A sliding-window
(SWA) KV cache covers the last 128 tokens. When the SWA cache is missing,
DeepInfer doesn't rebuild it exactly. It recomputes only the last 128 tokens and
restricts their attention to the recomputed segment. If replay starts at
position `s`, query `i` sees only keys `max(s, i−127) … i`. The doc calls the
result an approximate SWA KV cache. In the CED (encoder/decoder) layout this
means decoder layers prefill only 128 tokens instead of 2,560, which DeepSeek
says nearly halves total prefill compute.

The doc doesn't say how approximate. DeepInfer only runs DeepSeek checkpoints
on SM100 GPUs or Ascend NPUs, so I couldn't measure it on their model. I ran
the same trick on a small open model with real sliding-window layers instead,
and compared it with exact inference.

## Why it's an approximation at all

One sliding-window layer with window `W` looks back `W−1` tokens. Stack them and
the lookback adds up. A token's KV at layer 2 depends on layer-1 outputs up to
`W−1` back, and those depend on embeddings up to `2(W−1)` back. To get layer
`ℓ`'s SWA KV exactly you need roughly `ℓ·(W−1)` tokens of history:

```
exact SWA KV at layer ℓ needs  R ≥ ℓ·(W−1) + 1  replayed tokens

DeepSeek decoder:  20 layers × 128 window  ≈ 2,560 tokens   (their number)
bounded replay:    128 + 6 (spec. backtrack) → 134, configured as 140
```

The 2,560 in their doc is exactly this product. Bounded replay keeps only the
window factor. Every replayed token except the last few is computed from a
truncated history: the first token sees only itself, the second sees two
tokens, and so on. Those wrong-ish states then feed the decode tokens' windows.

## The experiment

**Model.** [Gemma-3-270M](https://huggingface.co/unsloth/gemma-3-270m) (base, fp32,
CPU). It has 18 layers: 15 sliding-window layers with `W = 512` and 3
full-attention layers (layers 6, 12, 18).

**Replay as a mask.** Bounded replay is just an attention-mask change, so I
didn't need a custom engine. Recent `transformers` versions accept a separate
mask per layer type. I run prompt and continuation in one teacher-forced pass:

```python
def masks(T, s=None):
    i, j = torch.arange(T)[:, None], torch.arange(T)[None, :]
    causal = j <= i
    local = causal & (j > i - W)                 # normal sliding window
    if s is not None:                            # bounded replay from s
        local = torch.where(i >= s, local & (j >= s), local)
    f = lambda m: torch.where(m, 0.0, NEG)[None, None]
    return {"full_attention": f(causal), "sliding_attention": f(local)}
```

Positions before `s` are computed exactly. In sliding layers, replayed tokens
and decode tokens can't see anything before `s`. The full-attention layers are
untouched. With `s = None`, the masks reproduce the stock forward pass
bit-for-bit (max |Δlogit| = 0.0), which confirms the harness is correct.

**Data.** Two 2,560-token prompts, each followed by 64 continuation tokens, from
each of *Moby-Dick*, *Pride and Prejudice* and RFC 9110. That gives 6 windows
and 384 "decode" positions per setting. I compare next-token distributions
with the exact run: mean KL(exact‖replay), top-1 agreement, and the worst
single step.

**The ratio that matches DeepInfer.** Their 140 kept tokens over a 128 window is
about 1.09·W. On Gemma that's `R = 560`.

## Results

| replay length R | R / W | mean KL (nats) | median-window KL | top-1 agree | worst step KL |
|---|---|---|---|---|---|
| 512 | 1.00 | 0.0063 | 0.0063 | 96.1% | 0.062 |
| **560** | **1.09** | **0.0054** | **0.0059** | **96.6%** | **0.095** |
| 640 | 1.25 | 0.0032 | 0.0032 | 98.4% | 0.047 |
| 768 | 1.50 | 0.0033 | 0.0022 | 97.9% | 0.304 |
| 1024 | 2.00 | 0.0034 | 0.0009 | 98.7% | 0.326 |
| 1536 | 3.00 | 0.0009 | 0.0007 | 98.7% | 0.016 |
| 2048 | 4.00 | 0.0007 | 0.0004 | 98.7% | 0.020 |

Those KL numbers mean nothing without a scale, so I added four reference points
(5 of the 6 windows; the sixth hit my time limit):

| reference | mean KL | top-1 agree |
|---|---|---|
| same model in **bf16** vs fp32, exact masks | 0.0009 | 98.4% |
| SWA cache **thrown away** (R = 0) | 0.83 | 58% |
| R = 128 (replay shorter than the window) | 0.084 | 85% |
| R = 384 | 0.016 | 94% |

Here's how I read these numbers:

1. **At DeepSeek's ratio, replay costs about 6× bf16's rounding noise.** At
   `R ≈ 1.09·W` the mean KL is 0.0054, against 0.0009 for running in bf16. Top-1
   agreement is 96.6%, against 98.4%. Greedy decoding picks a different token
   about once every 30 steps, compared with once every 60 from bf16 alone. That's
   real but small, and two orders of magnitude better than dropping the cache.

2. **The window requirement matters a lot.** DeepInfer asserts that the kept
   suffix covers the window plus the backtracking length
   ([`model_runner.py`](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/model_runner.py#L110)).
   Going from `R = W/4` to `R = W` cuts KL by 13×. Above `W` the error keeps fading. The
   median-window KL drops 7× from `W` to `2W` and about 2× more by `4W`. It reaches
   bf16-noise level around `R ≈ 2–3·W`, which is still far below the
   `layers × W` needed for exactness.

3. **The average hides a heavy tail.** Mean KL isn't monotone in `R`. At
   `R = 768` and `R = 1024`, one window each (a *Pride and Prejudice* window and
   an RFC window) has a single decode step with KL around 0.3 nats. Every other
   step in those windows is near zero. The replay boundary landed somewhere the
   model cares about, and one step's distribution moved a lot. If you validate
   an approximation like this with a mean over a benchmark, you won't see those
   steps.

4. **No measurable quality loss.** The change in NLL on the true continuation
   ranges from −0.002 to +0.015 nats per token across settings. That's inside
   the noise of 384 tokens. Replay makes outputs *different*, not detectably
   *worse*. This is the same kind of argument behind accepting batch-variant
   kernels, which DeepInfer also lets you switch off.

## Where the error lives

I also recorded the final-layer hidden-state error at each replayed position,
relative to the exact run (mean over windows, `R = 1024`):

```
offset from s:   0     64    128   256   384   512   768   1023
rel. error:    1.11  0.30  0.17  0.15  0.09  0.055 0.033  0.024
```

The first replayed token is completely wrong (relative error above 1). It sees
only itself, so in local layers it behaves like the start of a document. Error
decays over the next few hundred tokens but levels off around 2%. On Gemma it
never reaches zero because the full-attention layers also read the replayed
positions, so some contamination leaks into the global path. DeepSeek's CED
decoder avoids this. Its block KV comes from the encoder's last hidden state,
which replay doesn't touch, so the long-range path stays exact. **This is the
biggest difference between my setup and theirs, and it makes my numbers an
overestimate.**

## A fix that didn't work

My guess was that the damage came from the truncated windows having no
attention sink. In normal inference the first few positions of a sequence see
BOS, and models lean on it heavily. So I tried a
[StreamingLLM](https://arxiv.org/abs/2309.17453)-style patch: replayed tokens
whose window is truncated may also attend to token 0.

It made things **worse** at every replay length up to `3W`, and was a tie at
`4W`. At `R = 512`, mean KL rose
from 0.0063 to 0.0139, and top-1 agreement fell from 96.1% to 94.8%. The reason
is that those positions never saw BOS in the exact run either, since their real
windows had scrolled past it. Adding a sink fixes a problem the model doesn't
have and introduces a new mismatch.

I also measured attention mass directly to see whether the first replayed
tokens become a *new* sink:

- **Global layers.** In 11 of 12 cases, decode queries put *less* weight on
  the first four replayed keys than in the exact run (typically ≈0.0001 vs
  ≈0.0005). The real sink stays
  at BOS, which gets about 30% of the mass in both runs.
- **Sliding layers.** Replayed queries put about 2× more mass on the boundary
  tokens than the exact run does. In one of 12 cases the boundary token
  (" sails") became a strong local sink, at 27× its exact mass. That window
  still had a below-median KL.

So a fake sink does form sometimes, but it doesn't predict the output error.
I can't yet explain the 0.3-nat spikes, and I'd rather say so than guess.

## Caveats

- **Different model.** Gemma wasn't trained to see truncated windows. DeepSeek
  may have trained V4.1 with this truncation; the doc doesn't say either way.
  If they did, the error on their model could be much smaller than mine.
- **Different geometry.** Their window is 128 over 20 decoder layers; mine is
  512 over 15 sliding layers plus 3 global ones. I compared at the same `R/W`
  ratio, but nothing guarantees the error scales that way.
- **Small sample.** 6 windows × 64 positions is enough to see the trend and the
  tail. It isn't enough to put tight confidence intervals on a 0.005-nat effect.
- **Teacher-forced.** I scored continuations from the source text, not sampled
  ones. Under free-running decoding, a 3–4% top-1 disagreement can push a
  generation onto a different path.

## The takeaway

Bounded replay replaces an `O(layers × W)` recompute with an `O(W)` one. On a
model never trained for it, at DeepSeek's `R/W` ratio, it costs about 6× the
divergence of bf16 rounding, with no measurable loss in likelihood. That's a
good trade for a cache-miss path. Two things to copy if you build something
similar. Keep the replay at least one full window long, because below that the
error rises by an order of magnitude. And measure the worst decode step, not
just the mean, because the error shows up in single steps that a mean hides.

*Code: the masking trick is the snippet above. The full harness is three short
scripts (sweep, calibration, attention mass) using `transformers` 5.20 and
torch 2.14. It runs on a 4-core CPU in a couple of hours.*
