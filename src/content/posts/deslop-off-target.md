---
title: "The de-slopped Qwen3-4B, measured on what it wasn't scored on"
description: A Qwen3-4B fine-tune trained out the em dashes, "not X, but Y" and other assistant tics, filtering its training data with a regex detector. I ran it against the stock model on 136 generations and looked at what that detector can't see. The targeted tics are gone. Paragraphs, sentence rhythm and comma use changed too, and ordinary question prompts bring the old formatting back.
date: 2026-10-21
draft: false
---

This week Erik Kannike released
[a Qwen3-4B fine-tune with the AI writing tics trained out](https://100toolkit.com/blog/deslop-model/)
(it was on the HN Show page). The headline number: a "slop score" of 121 per
1000 words on the stock model drops to 12.5, below the 15 the author treats as
typical of human writing. Em dashes go from 20.5 per 1000 words to zero.

The method matters for everything below. Each training round, a teacher wrote
answers to 288 prompts. Only answers that scored under the threshold on a
26-trait regex detector, and passed some damage and genre checks, were kept.
A rank-16 LoRA was then trained on the survivors. That is rejection sampling
against the metric, so the metric will improve almost by construction. The
interesting question is what happens to everything the detector *doesn't*
count. Goodhart's law predicts the pressure goes somewhere.

The [model card](https://huggingface.co/erikkannike/qwen3-4b-deslop-v4) is
candid about some of this. It says the voice is "plain to the point of flat",
that output runs long, and that long outputs can loop. I wanted numbers for
those, plus a look for side effects nobody has listed yet.

## Setup

- **Models:** `qwen3-4b-deslop-v4-Q4_K_M.gguf` and the stock
  `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, both run with llama-cpp-python on a
  4-core CPU (about 7 tokens/s).
- **Sampling:** the card's recommended settings for *both* models.
  Temperature 0.7, top-p 0.9, repeat penalty 1.1 over the last 256 tokens, no
  system prompt.
- **Prompts:** 24 short prompts in four genres (explain, opinion, email,
  personal reflection), two seeds each. Four 700-word prompts, one seed each.
  Then 8 prompts phrased the way the author's eval phrases them ("Write a
  LinkedIn post about…", "Write an explainer about…"), two seeds each.
  That's 68 generations per model, about 20k words each.
- **Short answers were capped at 520 tokens.** The stock model hit that cap 22
  times and the fine-tune 6 times. Every number below is a per-1000-word rate
  or a per-answer ratio, so the truncation shouldn't distort them much.

The author's detector (`slopscore.py`) isn't public, so I wrote my own
approximation of the traits the post names: em dashes, "not X, but Y",
rhetorical questions answered right away, signposts, "delve"/"seamless". I
call these **on-target**. Everything else is **off-target**: features I chose
because the post doesn't say it measures them. It's about 90 lines of regex
and `statistics`. The core:

```python
OFF = {
    "en_dash_or_spaced_hyphen": r"–|\s-\s",
    "colon_in_prose":           r"(?<=[a-z]):\s",
    "parenthesis":              r"\(",
    "comma_which":              r",\s+which\b",
    "markdown_struct":          r"(?m)^\s*(?:#{1,6}\s|[-*]\s|\d+\.\s)",
    ...
}
# sentence rhythm: coefficient of variation of sentence length
d["sent_len_cv"] = pstdev(lens) / mean(lens)
# paragraphing
paras = re.split(r"\n\s*\n", text.strip())
d["words_per_para"] = len(words) / len(paras)
```

## 1. The targeted tics really are gone

On my approximation the claim holds up, including on prompts the author never
saw:

| per 1000 words | stock | de-slop |
|---|---|---|
| on-target total | 18.9 | 1.8 |
| em dashes ("Write …" prompts) | 14.3 | 0.6 |
| "not X, but Y" ("Write …" prompts) | 3.4 | 0.0 |
| answers under 15, author-style prompts | 2 / 16 | **16 / 16** |

My stock-model em dash rate (14–16 per 1000) is a little below the author's
20.5. That fits a different prompt set and a looser regex. The fine-tune does
use an em dash now and then, which the author's 0.0 doesn't show.

The suppression also spread to punctuation the detector doesn't target. I
expected the em dash to be swapped for en dashes, colons or semicolons. That
didn't happen; on "Write …" prompts all of them fall together:

| per 1000 words, "Write …" prompts | stock | de-slop |
|---|---|---|
| en dash / spaced hyphen | 3.0 | 0.0 |
| colon mid-sentence | 7.3 | 0.6 |
| semicolon | 1.1 | 0.1 |
| parenthesis | 1.1 | 0.0 |
| **all commas** (whole corpus) | **63.2** | **28.4** |

The last row is the one to notice. The fine-tune uses fewer than half as many
commas per word. It isn't swapping one punctuation mark for another. It
dropped almost all punctuation other than the full stop.

## 2. Where the pressure went: rhythm and paragraphs

With fewer commas and clauses, sentences get flatter. I measured how much
sentence length varies within an answer (the coefficient of variation, often
called "burstiness"). For reference I ran the same function over 300-word
chunks of human text:

| text | sentence-length CV |
|---|---|
| *Pride and Prejudice* (mid-book) | 0.79 |
| *Moby-Dick* (mid-book) | 0.76 |
| Paul Graham, "How to Do Great Work" | 0.49 |
| Paul Graham, "How People Get Rich Now" | 0.63 |
| stock Qwen3-4B, short answers | 0.59 |
| **de-slop, short answers** | **0.37** |
| **de-slop, 700-word pieces** | **0.29** |

The stock model is inside the human range I sampled. The fine-tune falls below
all of it. It writes sentences of nearly the same length, one after another.
Low burstiness is one of the oldest signals statistical AI-text detectors
look for, so on this one axis the de-slopped model looks *more* machine-made
than the original. Four human texts is a small reference set. Still, the
direction is clear, and the gap (0.29 against 0.49 at the low end) is large.

Paragraphs went the same way:

| | stock | de-slop |
|---|---|---|
| words per paragraph, "Write …" prompts | 39 | 152 |
| answers that are a single paragraph, "Write …" prompts | 0 / 26 | **15 / 26** |
| single-paragraph answers, LinkedIn prompts | 0 / 4 | 3 / 4 |
| single-paragraph answers, reflections | 0 / 12 | 9 / 12 |

A one-paragraph LinkedIn post is a wall of text. My guess at the mechanism:
the detector counts "short fragments used for emphasis", and in assistant
writing those usually stand alone as one-line paragraphs. Filtering them out
also removed the habit of breaking paragraphs at all. That is my inference.
I can't check it without the detector or the training set.

The flat, diary-like voice the card mentions can be counted too. In the
reflection prompts, **56% of the fine-tune's sentences start with "I"**,
against 12% for the stock model. Across the whole corpus, "I" is the
fine-tune's most common sentence opener (18.7% of sentences), followed by
"The" (11.6%). Typical run: "I moved into a small apartment… I set up a desk…
I would sit at my desk…"

Clauses that used to be joined with dashes and colons now get chained instead:

| per 1000 words | stock | de-slop |
|---|---|---|
| ", which" | 1.06 | 2.44 |
| ", so" | 0.25 | 1.35 |

The author says round 2 had a runaway ", which" problem and that round 3
filtered for it. In my samples it is still 2.3× the stock rate, and ", so" is
5.4×. The rates are low in absolute terms, but they are the only connectors
I measured that went *up*.

## 3. Ask a question and the assistant comes back

This was the result I didn't expect. The author's eval and, as far as the
post says, the training prompts are phrased as writing tasks ("Write a blog
post about…"). Real users often just ask questions. I split my prompts by
phrasing:

| answers with Markdown lists, headers or bold | stock | de-slop |
|---|---|---|
| prompts beginning "Write …" | 5 / 26 | 0 / 26 |
| "Write an explainer about X" | 4 / 4 | 0 / 4 |
| **"How does…/Why do…/What does…/Explain…"** | **12 / 12** | **10 / 12** |

The explainer genre is the clearest case. Asked to "Write an explainer about
how noise-cancelling headphones work", the fine-tune writes plain prose. Asked
"What does a database index actually do?", it writes `### 1. **It creates a
lookup table**` headers, a `---` rule, an "Important Notes:" list, and once a
👉 emoji and "So in short:" followed by an em dash. On these question-phrased
explain prompts the on-target score is 4.8 per 1000 words, against 13.0 for
the stock model. That's a real drop, but nowhere near the 1.2 on "Write …"
prompts. Opinion prompts phrased as questions ("Should schools still teach
cursive?") did *not* leak, so it isn't questions in general. It's the
"explain this to me" request, which is the most assistant-shaped one.

That would fit how the training worked, if the training prompts look like the
eval prompts (I haven't seen them). Then a LoRA with 161 examples learned
"when asked to write a piece, write like this", not "never write like an
assistant". The assistant style is still there, behind a different kind of
prompt.

## 4. Long outputs: no loops, but more repetition

With the repeat penalty on, none of the four 700-word pieces looped. They did
repeat more. In the fine-tune's pieces, 1.1% of 4-grams are repeats, against
0.27% for the stock model. Two of the stock model's pieces had no repeated
4-gram at all. Most of the repeats are habits of phrase: "you don't need to"
recurs through the running guide, and "he did not tell anyone about" through
the lighthouse story. Sentence-length CV in long pieces fell to 0.20–0.40
(stock: 0.38–0.67). I didn't test without the penalty. The card already warns
about that case.

## What to make of it

None of this means the project failed. The model does what it claims: on the
traits it was filtered for, the drop is large and holds on new prompts. What
the off-target numbers show is that "slop" was never just a list of tokens.
Press on the listed tokens and you get:

- fewer commas,
- more uniform sentences,
- no paragraph breaks,
- "I" at the start of every other sentence,

and none of those show up in the score being optimized. If you're building
something similar, the cheap fix is to add these as damage checks, the same
way the author added ", which" after round 2. Sentence-length CV and
words-per-paragraph are a few lines each. Add question-phrased prompts to
both the training and eval sets too. Otherwise the most common way people
actually prompt a model stays untested.

**Caveats.** I ran Q4_K_M quantizations of both models, not the author's bf16
merge (the Q4 file is the one the card recommends for laptops). My detector is
an approximation, not `slopscore.py`. Most cells are 8–26 samples, and the
long-form numbers rest on four pieces per model. The off-target effects are
large enough that I don't expect them to vanish with more samples. The exact
figures will shift.

*Sources:* [the author's write-up](https://100toolkit.com/blog/deslop-model/),
the [model card](https://huggingface.co/erikkannike/qwen3-4b-deslop-v4) and
[GGUF repo](https://huggingface.co/erikkannike/qwen3-4b-deslop-v4-GGUF), the
stock model via
[lmstudio-community's GGUF](https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF).
Human reference texts are from
[Project Gutenberg](https://www.gutenberg.org/) and
[paulgraham.com](https://paulgraham.com/). All generations and measurements
are my own.
