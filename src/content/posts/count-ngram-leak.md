---
title: "COUNT at toy scale: when the n-gram table already knows the answer"
description: "I rebuilt COUNT's n-gram input embeddings on a CPU. They work, but if you count the table on your own training text, the model learns to cheat. Leave-one-out counting fixes it."
date: 2026-10-21
draft: false
---

[COUNT](https://arxiv.org/abs/2610.20456) (Magnusson, Smith and Koh; UW and AI2,
posted 14 Oct) adds n-gram statistics to a transformer without adding any
learned parameters. You count n-grams over a corpus offline and keep, for each
history, the top-K next tokens with their
[stupid-backoff](https://aclanthology.org/D07-1090/) scores. At each position
you look up the current suffixes and build an extra input embedding from them:

```
f_t = (1/J) · Σ_j Σ_k  S(w_k | h_{t,j}) · B[w_k]      # B = the LM's own unembedding matrix
h_t^(0) = LayerNorm(E[x_t] + f_t)                     # "embedding amplification"
```

The table is read-only, so it can live in host RAM. The paper's largest table
is 797 GB. At 60M–1B parameters they report validation loss that a plain
transformer needs 46–89% more compute to match. They also report that you can
swap in a domain table at inference time with no training.

I can't check claims at 1B scale on a 4-core CPU. What I can do is build the
mechanism small and stress the parts that are easy to get wrong when you
reimplement it. One of those parts turned out to matter a lot: **which text the
table is counted on.**

## Setup (my experiment, not the paper's)

- **Data.** The [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  validation file, character level (72 symbols), split into three disjoint
  pieces. T is the 4M-character training text. D is a separate 4M characters
  used only for counting. V is 0.5M characters held out for validation. For
  the out-of-domain test I used
  [Tiny Shakespeare](https://github.com/karpathy/char-rnn): the first 1M
  characters for a table, the last 115K for evaluation.
- **Model.** A 4-layer, 128-wide, 4-head GPT with untied input embeddings
  `E` and output matrix `B`, and 128-character context. Each run trains for
  3,000 AdamW steps at 32×128 characters per step, about 3 passes over T.
  One run takes about 6 minutes on CPU.
- **Table.** History lengths {0, 1, 2, 3, 5, 8, 12} characters (J = 7), K = 8,
  and a backoff factor of 0.4 for histories missing from the table. I picked
  the 12-character maximum to roughly match the reach of the paper's 6-token
  histories. COUNT arms get embedding amplification, as in the paper.
- **Metric.** Mean cross-entropy in nats per character.

Training, counting and evaluation are all vectorized: rolling 64-bit hashes of
each history, `np.unique` for counts, and per-row top-K via `lexsort`.

## Result 1: COUNT works, and beats late fusion in-domain

| arm (validation on V) | nats/char |
|---|---|
| baseline transformer (seed 0 / seed 1) | 0.934 / 0.943 |
| baseline + late fusion with table D (best λ = 0.1) | 0.925 |
| **COUNT, table counted on D** (seed 0 / seed 1) | **0.838 / 0.842** |

The seed-to-seed spread is about 0.01, and COUNT's gain is about 0.10, roughly
10× the noise. So the paper's main qualitative claim holds in a setting nothing
like theirs: an n-gram table fed through the input beats the same table fed
into the output. Late fusion is the kNN-LM/infini-gram-style mix
`(1-λ)·p_model + λ·q_ngram`. At this scale it helps only a little, and from
λ = 0.3 upward it hurts.

## Result 2: count the table on the training text and the model learns to cheat

The paper counts its table on Dolma 3, the corpus it pretrains from. A natural
small-scale reimplementation does the same and counts the training text
itself. Here is what happens:

| table source | train loss (on T) | val loss (on V) |
|---|---|---|
| none (baseline) | 0.921 | 0.934 |
| **T itself, min history count 1** | **0.391** | **1.010** |
| T itself, min history count 16 | 0.871 | 0.907 |
| D (disjoint), min history count 1 | 0.817 | 0.838 |
| D (disjoint), min history count 16 | 0.900 | 0.910 |

Counted on its own training text with no threshold, COUNT reaches a training
loss of 0.39, less than half the baseline's. It also has the worst validation
loss of any arm, worse than using no table at all.

The cause is easy to see in the counts. For each training position I checked
how often its history appears in T:

```
history length   positions whose history occurs once   occurs < 16 times   val histories found in table
      5                       0.9%                           7.9%                    99.2%
      8                      10.0%                          38.1%                    90.6%
     12                      36.0%                          73.5%                    65.9%
```

For 36% of training positions, the 12-character history occurs exactly once.
That single occurrence is the current position, so the table row holds one
candidate with score 1.0, and that candidate is the correct next character. The
input embedding is mostly `B[target]`. Measured over all positions, the table's
top candidate is right **82.9%** of the time on T but **73.0%** on V. The
backbone learns to trust long-history rows at whatever reliability they show
during training. On held-out text, a third of positions have no 12-character
match and the matches that do exist are less reliable, so that trust fails.

Giving the leaky model the clean disjoint table at evaluation doesn't help:
validation goes to 1.034. The table at evaluation isn't the problem. The model
learned the wrong prior during training.

**Does this apply to the paper?** Probably very little, and I want to be clear
about why. Their main table keeps only histories seen at least 265 times. It
counts the full Dolma 3 corpus (trillions of tokens), and the models train on
at most about 40B tokens drawn from it. Any one training position therefore
contributes at most 1/265 of any row it lands in. My count-16 arm shows that a
threshold alone shuts the leak. But anyone who copies the method at small
scale, or builds a low-threshold table from a fine-tuning set and then trains
on that same set, will hit this. The training curve will look great, so
nothing will warn them. Their domain tables use thresholds of 4–45, but they
use those only at inference, so the issue doesn't come up there.

## Result 3: leave-one-out counting fixes it, and you keep the rare histories

A threshold fixes the leak by throwing away the rare, long histories that carry
the most information. The disjoint table at min count 16 scores 0.910, against
0.838 at min count 1. An old n-gram-LM trick avoids that trade:
**leave-one-out**. When building the feature for training position t, subtract
t's own occurrence from the counts before scoring:

```python
# per order j, for training positions only
hc_loo = C(h) - 1
c_loo  = C(h, w) - (w == x[t+1])     # remove this position's own continuation
hit    = hc_loo >= cmin               # a history seen only here is now "unseen" → back off
S      = c_loo / hc_loo
```

Validation and inference use the normal table. Only the training-time features
change, and the change is a few lines in the lookup.

| arm | table top-1 on T / V | train loss | val loss |
|---|---|---|---|
| self-counted, min count 1 | 82.9% / 73.0% | 0.391 | 1.010 |
| **self-counted, min count 1, leave-one-out** | **72.5% / 73.0%** | 0.802 | **0.832** |
| disjoint, min count 1 | 71.9% / 72.8% | 0.817 | 0.838 |

With leave-one-out, the table is as reliable on training text as on held-out
text, and the self-counted table scores as well as a disjoint one (0.832 vs.
0.838, within seed noise). Every row stays in, and you don't need a second
corpus. It's the same leave-one-out idea Ney, Essen and Kneser used to
estimate their discounts: a count shouldn't get credit for the data point it
is being asked to predict.

## Result 4: table swapping helps, but late fusion is better far out of domain

The paper's other selling point is swapping in a domain table at inference
time. I took the COUNT model trained with the TinyStories table and evaluated
it on Shakespeare, about as far from TinyStories as I could get:

| on Shakespeare (val) | nats/char |
|---|---|
| baseline | 3.004 |
| baseline + late fusion, Shakespeare table, λ = 0.5 | 1.838 |
| COUNT, TinyStories table | 2.881 |
| COUNT, 3:1 mix of TinyStories : Shakespeare scores (the paper's λ = 0.25) | 2.569 |
| COUNT, swapped to Shakespeare table (seed 0 / seed 1) | 2.248 / 2.338 |
| **COUNT, swapped table + late fusion, λ = 0.5** | **1.735** |

The swap works without any training: 2.88 → 2.25. But COUNT's input path loses
to late fusion on a plain baseline (1.84). That reverses the in-domain result.
My reading: COUNT never feeds n-gram scores straight into the output. They go
through a backbone that learned to map `f_t` into predictions on TinyStories
contexts, and on Shakespeare that mapping is out of distribution. Late fusion
injects probability mass directly, so it doesn't depend on how well the
backbone transfers. The paper's domains (math, biomedicine, law) are much
closer to the web data they pretrained on, and there they find a 1:3 mix best.
In my far-off case the full swap beats the mix. The best result here comes
from using both: the swapped table as input and as a late-fusion prior.

## What I'd take from this

1. **COUNT's core claim survives a ~1,000× scale-down.** Feeding count
   statistics through the input, weighted unembedding rows, cut validation loss
   by about 10% for a few lines of code and a hash table.
2. **The table is training data, so treat it as training data.** A table
   counted on the same text you train on, with a low threshold, gives the
   answer to the model. Use a high threshold as the paper does, a disjoint
   counting corpus, or (cheapest and best here) leave-one-out counts at
   training time.
3. **For a large domain shift, keep late fusion.** Swapping tables works, but
   the input path relies on a backbone that may not transfer. Adding the swapped
   table as an output prior too cost nothing and gave the best number in this
   experiment.

Caveats: this is one small character-level model and one corpus. Each arm has
one seed, except the baseline and disjoint-table arms, which have two. I
trained for multiple epochs where the paper trains for one, and the backbone
can only learn to rely on the leak because it sees the same positions again.
The leak itself, a table row whose only evidence is the current position, is
present from the first step. All numbers are from my own runs; the
paper-reported figures are cited as theirs.

*Sources: Magnusson, Smith & Koh,
[The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation](https://arxiv.org/abs/2610.20456)
(code: [IanMagnusson/COUNT](https://github.com/IanMagnusson/COUNT));
Brants et al. 2007, [stupid backoff](https://aclanthology.org/D07-1090/);
Eldan & Li, TinyStories; Karpathy, Tiny Shakespeare.*
