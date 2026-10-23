---
title: "Softmax Collapse, measured in PyTorch"
description: "PyTorch's cross-entropy returns an exactly zero target gradient once the logit margin passes about 16 in fp32, 7.7 in fp16 and 5.5 in bf16. On the paper's grokking task, Adam's eps=1e-7 removed the loss spikes as well as the new loss did, and label smoothing stopped grokking altogether."
date: 2026-10-23
draft: false
---

When a model is confident enough, `F.cross_entropy` sets the gradient on the
correct class to exactly zero and leaves the other gradients alone. In fp32 that
happens at a logit margin of about 16. In bf16 it happens at 5.5.

## Context

[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642)
(Barron, Gnanapragasam and Tschannen, Google DeepMind, October 2026) names two
numerical bugs in the standard softmax and cross-entropy. This post is about the
first one, which the paper calls Softmax Collapse.

The **logit margin** m is how far the correct class's logit sits above the
log-sum-exp of all the other logits. The exact cross-entropy loss is
log(1 + e^−m), and the exact gradient on the correct logit is about −e^−m. It
shrinks, but it never reaches zero.

Standard implementations subtract the max logit and sum the exponentials. The
max contributes exactly 1.0 to that sum and everything else adds up to about
e^−m. Once e^−m is smaller than half the gap between 1.0 and the next float,
the sum rounds to 1.0. From there on, the loss is 0 and the correct-class
gradient is 0. The other classes' gradients are computed without adding them
to 1.0, so they stay positive. The gradients stop summing to zero.

The paper argues this is behind the "Slingshot" loss spikes seen when training
with Adam, and grokking runs that stall. Its fix computes the margin while
leaving the correct class out of the sum, then returns softplus(−m). It gives
the cliffs as about 5.5 (bf16), 16.6 (fp32) and 36.7 (fp64), and its code is
JAX.

## The question

The paper answers "is this real?" Engineers who train with PyTorch want to know
something narrower:

1. Where does PyTorch's own `F.cross_entropy` hit the cliff? What about fp16,
   which the paper doesn't test, and `torch.autocast`, which is how most people
   use bf16?
2. On the paper's grokking task, does the fix do better than knobs you already
   have: Adam's ε, label smoothing, or casting the loss to fp64? The paper
   predicts the ε result in one sentence and doesn't run it.
3. Does an ordinary classifier ever get that confident?
4. What does the fix cost? The paper says "without meaningful slowdown" but
   gives no timings.

## Setup

Everything ran on a 4-core Xeon CPU with no GPU, using PyTorch 2.14.1. The
measured experiments took about 50 minutes of wall-clock time.

**The port.** This is the paper's hard-target cross-entropy, translated line by
line:

```python
def fixed_ce(z, t):
    zt = z.gather(-1, t[:, None])                     # correct-class logit
    is_t = F.one_hot(t, z.shape[-1]).bool()
    others = torch.where(is_t, float("-inf"), z - zt)  # leave it out of the sum
    m = -torch.logsumexp(others, -1)                   # logit margin
    return F.softplus(-m).mean()                       # log(1 + e^-m)
```

I also wrote the same math as a custom `torch.autograd.Function`, so the
backward pass is a single expression. The two agree to within 7e-9 in fp32.

**Cliff sweeps.** A single row of K logits, with the correct one set so the
margin is exactly m. I swept m and recorded the gradient that autograd
returns for the correct class.

**Grokking.** This follows the paper's description: a 1-layer, width-64 ReLU
transformer (no LayerNorm) learning a + b mod 31. Training uses a random 50%
of pairs, full batches, AdamW with betas (0.9, 0.98) and weight decay 1.0, for
10,000 steps, with 5 seeds per arm. One deviation: the learning rate is 3e-3.
At the paper's 1e-3, my model was still at 52–72% test accuracy at step 10,000
(2 seeds per loss), so there was no grokking to compare. The arms:

- `F.cross_entropy` (standard) vs fixed, fp32, ε = 1e-8
- standard with ε = 1e-7, with label smoothing 0.01, and with the logits cast to fp64 before the loss
- standard under `torch.autocast(bfloat16)`
- fp32 weights, but the loss computed on bf16 logits, standard vs fixed
- the whole model in bf16, standard vs fixed

I count a **loss spike** when the exact train loss jumps more than 10× above its
minimum over the previous 500 steps. I fixed that rule before the full grid ran.

**An ordinary classifier.** sklearn's bundled 8×8 digits (1,797 images): an
MLP with two 256-unit hidden layers, full-batch AdamW, 20,000 steps, weight
decay 0 or 0.01, 3 seeds, standard vs fixed.

## Results

### Where PyTorch's cliff is

<div id="sc-cliff" style="width:100%;min-height:400px" role="img" aria-label="Gradient on the correct-class logit as computed, divided by its exact value, versus logit margin from 0 to 24. The fixed cross-entropy stays at exactly 1. F.cross_entropy wobbles and then drops to exactly 0: bf16 at a margin of 5.55, fp16 at 7.65, fp32 at 16.0."></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function () {
  var C = {'F.cross_entropy, fp32': '#E69F00', 'F.cross_entropy, fp16': '#D55E00', 'F.cross_entropy, bf16': '#CC79A7', 'fixed CE, fp32': '#0072B2'};
  function theme() {
    var cs = getComputedStyle(document.documentElement);
    return {fg: cs.getPropertyValue('--fg').trim(), muted: cs.getPropertyValue('--muted').trim(), grid: cs.getPropertyValue('--line').trim()};
  }
  function axis(t, extra) { return Object.assign({gridcolor: t.grid, linecolor: t.muted, zerolinecolor: t.muted, tickfont: {color: t.muted}}, extra); }
  function draw(d) {
    var el = document.getElementById('sc-cliff'), t = theme(), narrow = el.clientWidth < 560, traces = [];
    ['fixed CE, fp32', 'F.cross_entropy, bf16', 'F.cross_entropy, fp16', 'F.cross_entropy, fp32'].forEach(function (k) {
      var r = d[k];
      traces.push({x: r.m, y: r.ratio, name: k, mode: 'lines', line: {color: C[k], width: k === 'fixed CE, fp32' ? 4 : 2},
        customdata: r.g.map(function (g, i) { return [g, r.exact[i]]; }),
        hovertemplate: 'margin %{x:.2f}<br>computed ÷ exact = %{y:.3f}<br>computed %{customdata[0]:.3e}, exact %{customdata[1]:.3e}<extra>' + k + '</extra>'});
    });
    var lay = {height: narrow ? 500 : 420, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', hovermode: 'closest',
      font: {family: 'Inter, system-ui, sans-serif', color: t.fg, size: 12},
      margin: {l: 56, r: 10, t: 10, b: narrow ? 150 : 100},
      legend: {orientation: 'h', x: 0, y: narrow ? -0.17 : -0.2, yanchor: 'top', font: {color: t.fg}},
      xaxis: axis(t, {range: [0, 24], dtick: 2, title: {text: 'logit margin m', font: {color: t.muted}, standoff: 6}}),
      yaxis: axis(t, {range: [-0.05, 2.05], title: {text: 'correct-class gradient, computed ÷ exact', font: {color: t.muted}}})};
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function go() {
    fetch('/lab/softmax-collapse-pytorch/cliff.json').then(function (r) { return r.json(); }).then(function (d) {
      draw(d);
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () { draw(d); });
      var el = document.getElementById('sc-cliff'), narrow = el.clientWidth < 560;
      window.addEventListener('resize', function () { if ((el.clientWidth < 560) !== narrow) { narrow = !narrow; draw(d); } });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
</script>
<script>
(function () {
  var C = {'F.cross_entropy, fp32': '#E69F00', 'F.cross_entropy, fp16': '#CC79A7', 'F.cross_entropy, bf16': '#D55E00', 'fixed CE, fp32': '#0072B2'};
  function theme() {
    var cs = getComputedStyle(document.documentElement);
    return {fg: cs.getPropertyValue('--fg').trim(), muted: cs.getPropertyValue('--muted').trim(), grid: cs.getPropertyValue('--line').trim()};
  }
  function axis(t, extra) { return Object.assign({gridcolor: t.grid, linecolor: t.muted, zerolinecolor: t.muted, tickfont: {color: t.muted}}, extra); }
  function draw(d) {
    var el = document.getElementById('sc-cliff'), t = theme(), narrow = el.clientWidth < 560, traces = [];
    ['fixed CE, fp32', 'F.cross_entropy, fp32', 'F.cross_entropy, fp16', 'F.cross_entropy, bf16'].forEach(function (k) {
      traces.push({x: d[k].m, y: d[k].g, name: k, mode: 'lines', connectgaps: false,
        line: {color: C[k], width: k === 'fixed CE, fp32' ? 5 : 2.5},
        hovertemplate: 'margin %{x:.2f}<br>|gradient| %{y:.3e}<extra>' + k + '</extra>'});
    });
    traces.push({x: d.exact.m, y: d.exact.g, name: 'exact (fp64 math)', mode: 'lines',
      line: {color: t.fg, width: 1.5, dash: 'dot'}, hovertemplate: 'margin %{x:.2f}<br>|gradient| %{y:.3e}<extra>exact</extra>'});
    var lay = {height: narrow ? 520 : 440, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', hovermode: 'closest',
      font: {family: 'Inter, system-ui, sans-serif', color: t.fg, size: 12},
      margin: {l: 64, r: 10, t: 10, b: narrow ? 170 : 110},
      legend: {orientation: 'h', x: 0, y: narrow ? -0.2 : -0.22, yanchor: 'top', font: {color: t.fg}},
      xaxis: axis(t, {range: [0, 40], title: {text: 'logit margin m (correct logit minus log-sum-exp of the rest)', font: {color: t.muted}, standoff: 6}}),
      yaxis: axis(t, {type: 'log', range: [-18, 0.1], exponentformat: 'power',
        title: {text: '|gradient on correct logit|', font: {color: t.muted}}})};
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function go() {
    fetch('/lab/softmax-collapse-pytorch/cliff.json').then(function (r) { return r.json(); }).then(function (d) {
      draw(d);
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () { draw(d); });
      var el = document.getElementById('sc-cliff'), narrow = el.clientWidth < 560;
      window.addEventListener('resize', function () { if ((el.clientWidth < 560) !== narrow) { narrow = !narrow; draw(d); } });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
</script>

*K = 32 classes with the 31 wrong logits equal, margin swept in steps of 0.05.
1.0 means the gradient is right, 0 means it's gone. Hover for the raw values.*

| `F.cross_entropy` input | cliff, PyTorch (31 equal wrong logits) | cliff, PyTorch (one runner-up) | paper |
|---|---|---|---|
| fp32 | 15.95 | 16.64 | 16.6 |
| bf16 | 5.53–5.58 | 5.55 | 5.5 |
| fp16 | 7.65 | — | not tested |
| fp64 | 34.66 | 36.74 | 36.7 |
| bf16 logits under `torch.autocast` | 16.05 | — | — |

Three things the paper doesn't show:

- **The cliff depends on how the wrong-class mass is spread.** When one
  runner-up holds all of it, PyTorch matches the paper's number. When the mass
  is spread over many classes, the cliff comes earlier: about ln 2 earlier for
  K ≥ 32, and ln 7 earlier at K = 8 (14.69). That pattern fits the summation
  order. At K = 8 each tiny term is added to the 1.0 on its own and lost on its
  own. At larger K, a tree reduction adds the 1.0 to only part of the small
  mass. That reading is my inference from the numbers. I didn't trace the
  kernel. `log_softmax` followed by `nll_loss` gives the same cliffs.
- **The gradient goes wrong before it reaches zero.** Near the cliff it can
  only take values that are whole multiples of the float spacing just below
  1.0, so it moves in coarse steps. In the margin range before the cliff,
  the computed value was as low as 0.53× the exact one in fp32. In fp16 and
  bf16 it ranged from 0.68× to 1.95×.
- **Autocast protects you.** Under `torch.autocast(bfloat16)` the logits come
  out as bf16, but autocast runs `F.cross_entropy` in fp32, so the cliff is
  fp32's. The bf16 cliff of 5.5 only applies if you call the loss on bf16
  logits yourself, for example in a model cast with `.to(torch.bfloat16)`.
  The same reasoning applies to fp16 and its 7.7.

The fixed loss had no cliff in any dtype I swept. In fp32 and fp64 its
correct-class gradient stayed nonzero up to a margin of 100, and its gradients
summed to zero within 4e-8. In fp16 and bf16 all the gradients eventually
underflow, but they do it together, not the correct class first.

### Grokking: the fix vs the knobs

Five seeds per arm, 10,000 full-batch steps. "Dip" is the worst drop in test
accuracy, after a run first reaches 90%.

| arm | final test acc, % (mean ± sd, min) | seeds ≥ 99% | spikes per run | worst dip, pts | final mean margin |
|---|---|---|---|---|---|
| standard, fp32 | 98.9 ± 1.9 (95.6) | 3/5 | 7.0 | 15.4 | 9.4 |
| **fixed, fp32** | **100.0 ± 0.0 (100.0)** | **5/5** | 0.4 | 1.9 | 17.0 |
| **standard, Adam ε = 1e-7** | **99.9 ± 0.2 (99.6)** | **5/5** | **0.0** | 0.4 | 14.7 |
| standard, label smoothing 0.01 | 46.6 ± 3.8 (42.2) | 0/5 | 0.0 | — | 4.6 |
| standard, loss in fp64 | 98.1 ± 3.6 (91.7) | 4/5 | 0.2 | 22.2 | 17.2 |
| standard, `autocast(bf16)` | 98.9 ± 1.6 (96.9) | 3/5 | 6.2 | 24.3 | 13.9 |
| standard, loss on bf16 logits | 97.2 ± 3.1 (92.7) | 4/5 | 40.8 | 50.9 | 4.3 |
| fixed, loss on bf16 logits | 99.2 ± 1.2 (97.3) | 4/5 | 0.0 | 1.5 | 17.2 |
| standard, whole model bf16 | 46.2 ± 3.5 (42.4) | 0/5 | 30.8 | — | 5.5 |
| fixed, whole model bf16 | 44.3 ± 2.9 (41.4) | 0/5 | 8.0 | — | 14.3 |

<div id="sc-curves" style="width:100%;min-height:560px" role="img" aria-label="Exact training loss (log scale) and test accuracy over 10,000 training steps for one seed at a time, comparing standard fp32 cross-entropy, fixed fp32 cross-entropy, standard with Adam epsilon 1e-7, and standard and fixed cross-entropy computed on bf16 logits. A seed selector switches between seeds 0 to 4."></div>
<p><label for="sc-seed">seed </label><select id="sc-seed"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></p>
<script>
(function () {
  var C = {std_fp32: '#E69F00', fixed_fp32: '#0072B2', 'std_fp32_eps1e-7': '#009E73', std_bf16logits: '#CC79A7', fixed_bf16logits: '#56B4E9'};
  var DASH = {fixed_bf16logits: 'dash'};
  var L = {std_fp32: 'standard, fp32', fixed_fp32: 'fixed, fp32', 'std_fp32_eps1e-7': 'standard, Adam ε = 1e-7',
    std_bf16logits: 'standard, loss on bf16 logits', fixed_bf16logits: 'fixed, loss on bf16 logits'};
  function theme() {
    var cs = getComputedStyle(document.documentElement);
    return {fg: cs.getPropertyValue('--fg').trim(), muted: cs.getPropertyValue('--muted').trim(), grid: cs.getPropertyValue('--line').trim()};
  }
  function axis(t, extra) { return Object.assign({gridcolor: t.grid, linecolor: t.muted, zerolinecolor: t.muted, tickfont: {color: t.muted}}, extra); }
  function draw(d) {
    var el = document.getElementById('sc-curves'), t = theme(), narrow = el.clientWidth < 560, traces = [];
    var seed = +document.getElementById('sc-seed').value;
    Object.keys(L).forEach(function (a) {
      var r = (d[a] || []).filter(function (x) { return x.seed === seed; })[0]; if (!r) return;
      traces.push({x: r.step, y: r.ce, name: L[a], legendgroup: a, mode: 'lines', line: {color: C[a], width: 2, dash: DASH[a] || 'solid'}, xaxis: 'x', yaxis: 'y',
        hovertemplate: 'step %{x}<br>train loss %{y:.2e}<extra>' + L[a] + '</extra>'});
      traces.push({x: r.step, y: r.test, name: L[a], legendgroup: a, showlegend: false, mode: 'lines', line: {color: C[a], width: 2, dash: DASH[a] || 'solid'}, xaxis: 'x2', yaxis: 'y2',
        hovertemplate: 'step %{x}<br>test accuracy %{y:.1f}%<extra>' + L[a] + '</extra>'});
    });
    var lay = {height: narrow ? 700 : 600, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', hovermode: 'closest',
      font: {family: 'Inter, system-ui, sans-serif', color: t.fg, size: 12},
      margin: {l: 60, r: 10, t: 10, b: narrow ? 190 : 120},
      legend: {orientation: 'h', x: 0, y: narrow ? -0.14 : -0.13, yanchor: 'top', font: {color: t.fg}},
      grid: {rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom', ygap: 0.12},
      xaxis: axis(t, {range: [0, 10000], matches: 'x2', showticklabels: false}),
      yaxis: axis(t, {type: 'log', exponentformat: 'power', range: [-8.5, 1], title: {text: 'exact train loss (nats)', font: {color: t.muted}}}),
      xaxis2: axis(t, {range: [0, 10000], title: {text: 'training step (full batch)', font: {color: t.muted}, standoff: 6}}),
      yaxis2: axis(t, {range: [0, 102], title: {text: 'test accuracy, %', font: {color: t.muted}}})};
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function go() {
    fetch('/lab/softmax-collapse-pytorch/grok-curves.json').then(function (r) { return r.json(); }).then(function (d) {
      draw(d);
      document.getElementById('sc-seed').addEventListener('change', function () { draw(d); });
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () { draw(d); });
      var el = document.getElementById('sc-curves'), narrow = el.clientWidth < 560;
      window.addEventListener('resize', function () { if ((el.clientWidth < 560) !== narrow) { narrow = !narrow; draw(d); } });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
</script>

*The loss is recomputed exactly in fp64 from the logits, so the cliff can't hide
it. Pick a seed. Every arm uses the same data split and initialization for a
given seed.*

Reading it:

- **The spikes reproduce, and they line up with the cliff.** Standard fp32
  cross-entropy produced 7 spikes in every one of its 5 runs. Before each
  spike, the mean margin had climbed to 15.9–16.9, where the loss rounds to
  zero. The peak losses after memorization were 0.12–0.72. Two seeds never
  reached 99% test accuracy.
- **The fix removes almost all of them.** All 5 seeds reached 100%. Two seeds
  still had one spike each, peaking at 0.013 and 0.025, which is about ten
  times smaller than standard's. The cliff can't explain those, so not every
  spike in this setup is Softmax Collapse.
- **Adam's ε = 1e-7 did as well as the fix here.** It produced zero spikes in
  5 of 5 seeds, and its worst final accuracy was 99.6%. The paper explains why
  in its Section 5. With weight decay, margins settle where the gradient is
  about the size of ε, near log(1/ε). That is 18.4 for ε = 1e-8, just past
  fp32's cliff, and 16.1 for ε = 1e-7, just before it. My ε = 1e-7 runs
  settled at a mean margin of 14.7. At no logged step were more than 5–9% of
  training examples past the cliff.
- **Label smoothing 0.01 caps the margin and kills grokking.** The margin
  can't go above about 4.6, so the cliff never comes into play. Test accuracy
  also never went above 42–52% (depending on the seed) in 10,000 steps.
- **Casting the loss to fp64** moves the cliff to 34.7, beyond the equilibrium.
  Four seeds had no spikes, though one of them was still at 91.7% at step
  10,000. The fifth had a single large spike (0.86) late in training. That's
  the same kind of non-cliff spike the fixed loss had, but bigger.
- **Autocast behaved like fp32**: 6–7 spikes per run, as the cliff table
  predicts.
- **The bf16 cliff is the destructive one.** With the loss on bf16 logits,
  standard cross-entropy spiked 38–46 times per run. In every run, test
  accuracy fell by 40–51 points at some point after first reaching 90%. The
  fixed loss on the same bf16 logits had no spikes and no dip above 1.5
  points. Four seeds per arm still reached 99%.
- **The whole model in bf16 didn't grok with either loss.** That's not the
  loss's doing. AdamW updates of size lr × 1 are mostly rounded away against
  bf16 weights. The paper reports fixed bf16 reaching 100%. It doesn't say how
  it stores the weights, and I didn't try fp32 master weights beyond the arm above.

One result goes against the paper's "groks faster". On the 3 standard-fp32
seeds that did reach 99%, they got there at steps 4,280–5,160. The fixed seeds
took 3,460–9,700. With 5 seeds and this spread, I can't call either one faster.

### An ordinary classifier

| digits MLP, 20k steps | test acc, % | median train margin at the end | training examples whose correct-class prob is exactly 1.0 in fp32 (peak) |
|---|---|---|---|
| standard, wd 0 | 96.9 | 29.1 | 97% |
| fixed, wd 0 | 97.1 | 32.0 | — |
| standard, wd 0.01 | 97.2 | 27.4 | 95% |
| fixed, wd 0.01 | 97.1 | 29.2 | — |

*Mean of 3 seeds. The last column is reported for the standard loss only. For
those examples, the correct-class gradient is zero.*

A plain MLP, trained long past 100% train accuracy, puts nearly every training
example past the fp32 cliff. In all four arms, the first examples cross by step
200. Nothing visibly breaks: no spikes, and test accuracy matches the fixed
loss within a few test images. The bug is everywhere in this run, and here it
doesn't hurt. Notice that the standard loss's margins still grew to 27–29 after
their correct-class gradients went to zero. The wrong-class gradients kept
pushing on their own, which is the asymmetry the paper describes.

### What the fix costs

Forward plus backward on CPU, 4 threads, median of 7 runs (`bench_speed.py`):

| logits | `F.cross_entropy` | fixed, plain autograd | fixed, `autograd.Function` |
|---|---|---|---|
| fp32, 1,024 × 32k | 100 ms | 370 ms (3.7×) | 171 ms (1.7×) |
| fp32, 512 × 128k | 204 ms | 759 ms (3.7×) | 303 ms (1.5×) |
| bf16, 1,024 × 32k | 58 ms | 279 ms (4.8×) | 170 ms (2.9×) |
| bf16, 512 × 128k | 115 ms | 519 ms (4.5×) | 317 ms (2.8×) |

A line-by-line port is 3.7–4.8× slower than the fused loss. A custom
backward brings that down to 1.5–2.9×. Most of the rest is the extra full-size
temporary (z − z_t with the target masked), which PyTorch's fused kernel never
materializes. These are CPU numbers. On a GPU, or with a fused kernel written for
the fix, the gap could be very different, and I didn't measure either.

## What it means

The mechanism is ordinary floating point. 1.0 plus anything smaller than half
a float spacing is 1.0. Cross-entropy builds a sum that contains a 1.0 by
construction, so the correct class's gradient is the first thing lost. What
decides whether that matters is the interaction with Adam. Adam divides by a
running estimate of gradient size, so a gradient that is exactly zero for a
while, and then isn't, turns into an oversized step.

Takeaways, with their limits:

- **If you use `torch.autocast`, the cliff you can hit is fp32's (about 16).**
  You need margins that large, which in practice means training long after
  memorization: small datasets, many epochs, distillation targets. The digits
  run shows that's easy to reach. It also shows reaching it can be harmless.
- **If you compute the loss on bf16 or fp16 logits, you're at 5.5 or 7.7.**
  A model that is 99.6% sure of the answer is already there in bf16. That was
  the worst arm I ran. Cast the logits to fp32 before the loss, or use the
  fixed loss.
- **On this task, Adam ε = 1e-7 was as good as the new loss, at no cost.** The
  catch is that ε is not a free knob at scale. [Wortsman et al.](https://arxiv.org/abs/2309.14322)
  found the default 1e-8 already too large for a 4.8B-parameter model,
  lowered it to 1e-15, and saw a divergence at 1e-6. Lowering ε pushes the
  equilibrium margin to log(1/ε), about 34.5 for 1e-15. That's far past the
  fp32 cliff, and there the fix is the only one of these options that helps.
- **Label smoothing is not a fix for this.** It avoids the cliff by never
  letting the model get confident, and on a grokking task that's the whole
  game.

Limits of this experiment: one toy task, one small transformer at a learning
rate 3× the paper's, 5 seeds, CPU only, 10,000 steps. The spike rule is mine.
The digits run is a single small model. I didn't test the paper's second bug
(Softmax Cancellation in distillation) or its no-weight-decay settings, where
it reports the largest accuracy collapses.

## Reproduce it

Code and raw results: [/code/softmax-collapse-pytorch/](/code/softmax-collapse-pytorch/)
([tarball](/code/softmax-collapse-pytorch.tar.gz)). `ce.py` has the PyTorch port.

```
python -m venv venv && venv/bin/pip install torch==2.14.1 numpy scikit-learn
PY=venv/bin/python ./run_all.sh
```

About 50 minutes on a 4-core CPU. The grokking grid takes most of it, and you
can rerun a single arm with `python grok.py fixed_fp32 0`, about 2.5 minutes.
