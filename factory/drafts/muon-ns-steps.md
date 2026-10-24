---
title: "Muon's Newton-Schulz steps, counted"
description: A new paper models Muon's Newton-Schulz steps as a damped polar update that can beat exact orthogonalization. On a small char-level GPT, 5 and 8 steps and an exact SVD tie within seed noise, 3 steps trail by under 0.01 nats, and the damped model tracks NS within about 0.008 nats, but damping an exact SVD didn't beat it.
date: 2026-10-24
draft: false
---

Muon, the optimizer, orthogonalizes every weight update with five iterations
of a matrix polynomial. A new paper argues those five steps aren't merely an
approximation of the ideal update. They are a different update, and on a
nearly rank-deficient toy problem a better one. I swept the step count in real training to see
whether that survives contact with a network.

## Context

[Muon](https://github.com/KellerJordan/Muon) treats each 2-D weight matrix's
momentum `M` as a matrix, not a bag of numbers. It replaces `M = U Σ Vᵀ` with
its polar factor `U Vᵀ`: same singular vectors, every singular value set to 1.
An SVD per matrix per step is slow on accelerators, so Muon normalizes `M` by its
Frobenius norm and runs a few Newton-Schulz (NS) iterations instead. Each
iteration is a fixed odd polynomial applied to the singular values. Muon's
coefficients are tuned to push values toward 1 fast rather than exactly, and
the default is 5 steps.

[A Quadratic Lens on Muon's Polar Update](https://arxiv.org/abs/2610.30280)
(Shulgin, Laing, Orvieto and Richtárik, October 2026) analyzes the exact polar
update on matrix quadratics. One of its results is about the tiny singular
values. Exact polar multiplies a singular value `s` by `1/s`, which is
unbounded. Finite NS multiplies small values by a fixed gain
`ℓ_q = 3.4445^q` (the polynomial's slope at zero, applied `q` times), so it
behaves like a **damped** polar update, `s ↦ s / √(s² + ε)` with `ε = ℓ_q⁻²`.
Below about `1/ℓ_q`, NS treats a direction like plain gradient descent does.
The paper reports that this can make fewer NS steps converge faster than exact
polar when the iterate is close to rank-deficient.

## The question

The paper's evidence for that sentence is a 10-dimensional quadratic with an
identity Hessian and 3–5 NS steps (its Appendix A.4). Its network evidence is
a diagnostic: in a 90M-parameter transformer trained with Muon, it reports
that about a quarter of the attention matrices' singular values sit below
`1/ℓ_5 ≈ 2·10⁻³`. Its conclusion suggests the step count is a design choice.
It doesn't train anything with a different step count.

So, in actual training:

1. Does the loss depend on the NS step count, and is exact polar worse than
   5 steps?
2. Does the paper's damped-polar model, with `ε = ℓ_q⁻²`, predict the loss of
   `q` NS steps? If yes, the quintic polynomial's other quirk (it leaves
   larger singular values anywhere in roughly 0.7–1.2 instead of at 1) doesn't
   matter.

## Setup

**Model and data.** A 4-layer, 4-head char-level GPT (width 128, context 64,
0.81M parameters) on
[tiny-shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare),
batch 32, 600 steps. The learning rate is constant, then decays linearly to
zero over the last 30%. Validation loss is the mean cross-entropy, in nats per
character, over 8 fixed batches from the held-out last 10% of the text.

**Optimizer.** Muon on the 16 weight matrices inside the blocks (fused QKV,
attention output, two MLP matrices per layer): Nesterov momentum 0.95, update
scaled by `√max(1, rows/cols)`, as in the reference implementation. Embeddings,
the output head and LayerNorms use AdamW at 3e-3 in every Muon arm. The only
thing that changes between arms is the function applied to the normalized
momentum:

```python
A, B, C = 3.4445, -4.7750, 2.0315           # Muon's quintic coefficients

def ns(G, q):                                # what Muon runs
    X = G / (G.norm() + 1e-7)
    for _ in range(q):
        S = X @ X.T
        X = A * X + (B * S + C * S @ S) @ X
    return X

def damped_polar(G, eps):                    # the paper's model of ns(G, q)
    U, s, Vh = torch.linalg.svd(G / (G.norm() + 1e-7), full_matrices=False)
    return (U * (s / torch.sqrt(s * s + eps))) @ Vh   # eps = A ** (-2 * q)
```

Both are simplified (the real code transposes tall matrices and runs the SVD
in float64). Exact polar is `damped_polar` with `eps = 0`. NS runs in fp32 here; the
reference implementation runs it in bfloat16.

**Fairness.** Each arm gets its own learning rate from a sweep over
{0.01, 0.02, 0.04, 0.08, 0.16} (AdamW: {1, 2, 4, 8}·10⁻³) on seed 0. Then 4
more seeds at that rate. Seeds fix the initialization and the batch order, so
two arms with the same seed see the same data. I report the mean and sample
standard deviation over the 5 seeds and the paired difference where it matters.
Seed 0 also picked the learning rates, a mild selection effect in its favor.
To check that longer training doesn't change the picture, 4 arms also ran for
2,000 steps with 2 seeds.

**Hardware.** A 4-core Xeon (2.8 GHz), no GPU, 4 single-threaded runs at a
time. Sweep and seeds: 37 + 41 minutes.

The effect needs tiny singular values, so I checked that they exist here.
At the end of training, 4.1–5.3% of the momentum buffers' normalized
singular values fall below `1/ℓ_5` (4 seeds each of the 5-step and the exact
arms). They concentrate in the attention output projection (10% on average,
up to 20% in one layer) and the second MLP matrix of the last layer (up to
25%). The regime exists here, just smaller than the paper's quarter of the
attention singular values. (Eight of the sixteen matrices also have one
singular value below 10⁻⁷, a structural zero, likely from LayerNorm; excluding
them lowers these shares by about 0.4 points.)

## Results

<div id="ns-filters" style="width:100%;min-height:420px" role="img" aria-label="Singular-value filters: output singular value versus input normalized singular value on a log axis from 1e-5 to 1, for Newton-Schulz with 1, 3 and 5 steps (solid) and the matching damped polar update (dashed), plus exact polar at 1."></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

*What each update does to one singular value of the normalized momentum.
Solid: NS with q steps. Dashed: damped polar with ε = ℓ_q⁻². Hover for values.*

The filters first. Both families agree near zero and differ on the ramp.
Past the ramp, damped polar sits at 1, while NS peaks at 1.2 and then
wobbles between about 0.68 and 1.13.
With one step the band is the whole spectrum: Frobenius normalization means
at most 11 of a matrix's singular values can exceed `1/ℓ_1 = 0.29`, so a
1-step Muon update is close to a rescaled momentum.

Final validation loss after 600 steps, 5 seeds each:

| update | LR | val loss, mean ± sd | range | optimizer ms/step |
|---|---|---|---|---|
| AdamW (reference) | 0.004 | 1.774 ± 0.010 | 1.763–1.789 | 7 |
| NS, 1 step | 0.04 | 1.796 ± 0.011 | 1.781–1.808 | 14 |
| NS, 2 steps | 0.02 | 1.733 ± 0.011 | 1.715–1.745 | 20 |
| NS, 3 steps | 0.02 | 1.693 ± 0.010 | 1.679–1.703 | 26 |
| NS, 5 steps (Muon default) | 0.02 | 1.686 ± 0.006 | 1.680–1.694 | 39 |
| NS, 8 steps | 0.02 | 1.684 ± 0.006 | 1.678–1.690 | 59 |
| exact polar (SVD) | 0.02 | 1.690 ± 0.011 | 1.677–1.706 | 93 |
| damped polar, ε = ℓ₁⁻² | 0.04 | 1.775 ± 0.012 | 1.763–1.794 | 91 |
| damped polar, ε = ℓ₃⁻² | 0.02 | 1.701 ± 0.014 | 1.688–1.721 | 96 |
| damped polar, ε = ℓ₅⁻² | 0.02 | 1.694 ± 0.010 | 1.684–1.705 | 97 |

The last column is the median time of the optimizer step (Muon plus the
AdamW part) on a shared CPU. Polar and damped arms use a float64 SVD. Read it
as a ranking, not as GPU numbers.

<div id="ns-final" style="width:100%;min-height:440px" role="img" aria-label="Final validation loss after 600 steps versus number of Newton-Schulz steps (1, 2, 3, 5, 8, exact), for Newton-Schulz and damped polar, with one dot per seed, exact polar at the right, and AdamW as a horizontal reference line."></div>

*Small dots: individual seeds. Diamonds: mean over 5 seeds (damped polar was
run at 1, 3 and 5 only). The dashed line is AdamW's
mean. Lower is better.*

Plainly:

- **Most of the gain is in the first 3 steps.** One step to two is worth
  0.063 nats, two to three 0.040, and both gains hold on every seed (paired).
  Three to five is 0.007 (5 steps ahead on 4 of 5 seeds), and three to eight
  is 0.009 (8 steps ahead on all 5, by 0.001–0.013). Small, but real by the
  same every-seed standard. Five to eight is 0.002.
- **Exact polar is not worse, and not better.** It is 0.003 behind 5 steps on
  average (paired sd 0.007; 5 steps ahead on 4 of 5 seeds) and 0.005 behind 8
  steps (3 of 5). That's within noise.
- **One NS step is worse than AdamW** (0.022 behind on every seed). Two steps
  already beat it by 0.041 on every seed.
- **The damped model predicts NS from 3 steps up.** Matched pairs differ by
  0.008 at `q = 3` and 0.007 at `q = 5`, with NS ahead on 4 of 5 seeds both
  times. At `q = 1` it misses: damped is 0.020 better on all 5 seeds, and
  lands on AdamW's number. Across step counts it gets the ranking right.
- **Damping an exact SVD didn't help.** Exact polar and damped polar at
  `ε = ℓ_5⁻²` differ only in `ε`. Exact polar was 0.004 ahead, on 4 of 5
  seeds (paired sd 0.004).

<div id="ns-long" style="width:100%;min-height:420px" role="img" aria-label="Validation loss versus training step over 2000 steps for Newton-Schulz with 3 and 5 steps, exact polar and damped polar with epsilon equal to ell_5 to the minus 2, mean of 2 seeds."></div>

*2,000-step runs, mean of 2 seeds. Hover for each step's values.*

<script>
(function () {
  var NS = '#0072B2', DAMP = '#E69F00', POLAR = '#009E73', ADAM = '#999999';
  var QC = {'1': '#CC79A7', '3': '#D55E00', '5': '#0072B2'};
  var LONG = {ns3: ['NS, 3 steps', '#D55E00', 'solid'], ns5: ['NS, 5 steps', NS, 'solid'],
    polar: ['exact polar', POLAR, 'solid'], damped5: ['damped polar, ε = ℓ₅⁻²', DAMP, 'dash']};
  function theme() {
    var cs = getComputedStyle(document.documentElement);
    return {fg: cs.getPropertyValue('--fg').trim(), muted: cs.getPropertyValue('--muted').trim(), grid: cs.getPropertyValue('--line').trim()};
  }
  function axis(t, extra) { return Object.assign({gridcolor: t.grid, linecolor: t.muted, zerolinecolor: t.muted, tickfont: {color: t.muted}}, extra); }
  function title(t, s) { return {text: s, font: {color: t.muted}, standoff: 6}; }
  function base(t, narrow, h, b) {
    return {height: h, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', hovermode: 'closest',
      font: {family: 'Inter, system-ui, sans-serif', color: t.fg, size: 12}, annotations: [], shapes: [],
      margin: {l: 56, r: 10, t: 34, b: b},
      legend: {orientation: 'h', x: 0, y: narrow ? -0.2 : -0.22, yanchor: 'top', font: {color: t.fg}}};
  }
  function drawFilters(d) {
    var el = document.getElementById('ns-filters'), t = theme(), narrow = el.clientWidth < 560, tr = [];
    var lay = base(t, narrow, narrow ? 560 : 440, narrow ? 190 : 120);
    lay.title = {text: 'One singular value, before and after', font: {size: 14, color: t.fg}, x: 0, xanchor: 'left'};
    lay.xaxis = axis(t, {type: 'log', range: [-5, 0], exponentformat: 'power', title: title(t, 'input singular value (normalized)')});
    lay.yaxis = axis(t, {range: [0, 1.25], title: title(t, 'output singular value')});
    lay.shapes.push({type: 'line', xref: 'x', yref: 'y domain', x0: d.inv_l5, x1: d.inv_l5, y0: 0, y1: 1, line: {color: t.muted, width: 1, dash: 'dot'}});
    lay.annotations.push({text: '1/ℓ₅', xref: 'x', yref: 'y domain', x: Math.log10(d.inv_l5), y: 1, yanchor: 'bottom', showarrow: false, font: {color: t.muted, size: 11}});
    tr.push({x: [d.s[0], d.s[d.s.length - 1]], y: [1, 1], mode: 'lines', name: 'exact polar', line: {color: POLAR, width: 2},
      hovertemplate: 'exact polar: every singular value → 1<extra></extra>'});
    ['1', '3', '5'].forEach(function (q) {
      tr.push({x: d.s, y: d.ns[q], mode: 'lines', name: 'NS, ' + q + (q === '1' ? ' step' : ' steps'), line: {color: QC[q], width: 2.5},
        hovertemplate: 'in %{x:.2e} → out %{y:.3f}<extra>NS, q=' + q + '</extra>'});
    });
    ['1', '3', '5'].forEach(function (q) {
      tr.push({x: d.s, y: d.damped[q], mode: 'lines', name: 'damped, ε = ℓ' + '₁₃₅'['135'.indexOf(q)] + '⁻²', line: {color: QC[q], width: 2, dash: 'dash'},
        hovertemplate: 'in %{x:.2e} → out %{y:.3f}<extra>damped polar, q=' + q + '</extra>'});
    });
    Plotly.react(el, tr, lay, {responsive: true, displayModeBar: false});
  }
  var ORDER = ['1', '2', '3', '5', '8', 'exact'];
  function drawFinal(d) {
    var el = document.getElementById('ns-final'), t = theme(), narrow = el.clientWidth < 560, tr = [];
    var lay = base(t, narrow, narrow ? 540 : 440, narrow ? 150 : 110);
    lay.title = {text: 'Validation loss after 600 steps', font: {size: 14, color: t.fg}, x: 0, xanchor: 'left'};
    lay.xaxis = axis(t, {type: 'category', categoryorder: 'array', categoryarray: ORDER, title: title(t, 'Newton-Schulz steps q')});
    lay.yaxis = axis(t, {title: title(t, 'val loss, nats/char (lower is better)')});
    lay.shapes.push({type: 'line', xref: 'x domain', x0: 0, x1: 1, yref: 'y', y0: d.adamw.mean, y1: d.adamw.mean, line: {color: ADAM, width: 1.5, dash: 'dash'}});
    lay.annotations.push({text: 'AdamW ' + d.adamw.mean.toFixed(3), xref: 'x domain', x: 1, xanchor: 'right', yref: 'y', y: d.adamw.mean, yanchor: 'bottom', showarrow: false, font: {color: t.muted, size: 11}});
    function series(name, color, arms, cats, off) {
      var xs = [], ys = [], hs = [];
      arms.forEach(function (a, i) { d[a].vals.forEach(function (v, j) { xs.push(cats[i]); ys.push(v); hs.push('seed ' + d[a].seeds[j] + ', lr ' + d[a].lr); }); });
      tr.push({x: xs, y: ys, mode: 'markers', name: name + ' (seeds)', legendgroup: name, showlegend: false, marker: {color: color, size: 6, opacity: 0.55},
        offsetgroup: name, text: hs, hovertemplate: 'q = %{x}: %{y:.3f} nats<br>%{text}<extra>' + name + '</extra>'});
      tr.push({x: cats, y: arms.map(function (a) { return d[a].mean; }), mode: name === 'Newton-Schulz' ? 'lines+markers' : 'markers', name: name, legendgroup: name,
        line: {color: color, width: 2.5}, marker: {size: 10, symbol: 'diamond'},
        hovertemplate: 'q = %{x}: mean %{y:.3f} nats (5 seeds)<extra>' + name + '</extra>'});
    }
    series('Newton-Schulz', NS, ['ns1', 'ns2', 'ns3', 'ns5', 'ns8'], ['1', '2', '3', '5', '8']);
    series('damped polar, ε = ℓ_q⁻²', DAMP, ['damped1', 'damped3', 'damped5'], ['1', '3', '5']);
    series('exact polar (SVD)', POLAR, ['polar'], ['exact']);
    Plotly.react(el, tr, lay, {responsive: true, displayModeBar: false});
  }
  function drawLong(d) {
    var el = document.getElementById('ns-long'), t = theme(), narrow = el.clientWidth < 560, tr = [];
    var lay = base(t, narrow, narrow ? 520 : 420, narrow ? 150 : 110);
    lay.title = {text: '2,000-step runs (mean of 2 seeds)', font: {size: 14, color: t.fg}, x: 0, xanchor: 'left'};
    lay.xaxis = axis(t, {range: [150, 2020], title: title(t, 'training step')});
    lay.yaxis = axis(t, {range: [1.5, 2.05], title: title(t, 'validation loss, nats/char')});
    lay.hovermode = 'x unified';
    ['ns3', 'ns5', 'polar', 'damped5'].forEach(function (a) {
      var runs = Object.keys(d[a]).map(function (k) { return d[a][k]; }), step = runs[0].step;
      var mean = step.map(function (_, i) { return runs.reduce(function (s, r) { return s + r.val[i]; }, 0) / runs.length; });
      tr.push({x: step, y: mean, mode: 'lines', name: LONG[a][0], line: {color: LONG[a][1], width: 2.2, dash: LONG[a][2]},
        hovertemplate: '%{y:.3f} nats<extra>' + LONG[a][0] + '</extra>'});
    });
    Plotly.react(el, tr, lay, {responsive: true, displayModeBar: false});
  }
  function load(u) { return fetch(u).then(function (r) { return r.json(); }); }
  function go() {
    Promise.all([load('/lab/muon-ns-steps/filters.json'), load('/lab/muon-ns-steps/final-loss.json'), load('/lab/muon-ns-steps/long-curves.json')]).then(function (v) {
      function all() { drawFilters(v[0]); drawFinal(v[1]); drawLong(v[2]); }
      all();
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', all);
      var el = document.getElementById('ns-final'), narrow = el.clientWidth < 560;
      window.addEventListener('resize', function () { if ((el.clientWidth < 560) !== narrow) { narrow = !narrow; all(); } });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
</script>

At 2,000 steps the four arms finish at 1.535–1.537 (individual seeds
1.529–1.543). The spread between seeds is larger than the spread between
updates.

## What it means

The paper's mechanism holds up in the part that's easy to check. Five quintic
NS steps act like polar with a floor on how hard they amplify tiny directions,
and an SVD with that floor trains about the same as NS. The polynomial's
wobble around 1 costs nothing: damped polar has none, and NS beat its damped
twin on 4 of 5 seeds at both 3 and 5 steps, by less than 0.01.

The surprising half doesn't show up at this scale. Exact polar trailed
5 steps by 0.003 on average, inside the seed noise, and 3 steps beat exact
polar on only 2 of 5 seeds. The cleanest test of damping itself, the same
SVD with and without `ε`, went slightly the other way.

My guess at why, which is my inference and not the paper's analysis: there
isn't much update for damping to take away. The paper reports that in its
90M model the directions below `1/ℓ_5` carry about 16% of the exact polar
update's squared norm, and about 5% of the 5-step one. In my runs that share
is 4.7% under exact polar and 1.7% under 5 steps. It grows only slightly
with training: at step 2,000, 4.7–6.5% of singular values sit below `1/ℓ_5`
across the four long-run arms. The paper's slow regime is defined by the
conditioning of the iterate on a quadratic, and its own diagnostic finds the
momentum's tiny directions unaligned with the weights' smallest ones, so the
share is a proxy, not its criterion.

Takeaways:

- If you run Muon, the default 5 steps is not leaving loss on the table
  against an exact SVD, at least at this scale. You don't need a better
  orthogonalizer for its own sake.
- Three steps cost 0.007–0.009 nats here (against 5 and 8 steps) at 3/5 of
  the NS work; two steps cost 0.046. If NS time matters, measure 3 or 4 before going lower.
- One step barely orthogonalizes anything (see the filter chart), and it lost
  to AdamW.
- To reason about a step count or a new coefficient set, the damped-polar
  formula is a decent stand-in from 3 steps up: compute `ℓ_q` and see where
  your momentum's singular values sit.

**Where this breaks.** This is a 0.81M-parameter character model, 600 and
2,000 steps, on a CPU. In the paper's analysis, exact polar's slowness grows with
the share of tiny singular values, and by update norm that share is about
three times larger in its 90M model than here. A larger model trained longer could still show fewer steps beating
exact. NS ran in fp32; in bfloat16 the tiniest values behave differently. QKV
is one fused matrix here, while the paper counts query, key and value as
separate matrices. The learning-rate grid is coarse (factors of 2), and I used one
learning rate for all seeds of an arm. I tested only the quintic Muon
coefficients, not tuned alternatives like Polar Express. And the timing column
is CPU wall-clock, which says little about a GPU.

## Reproduce it

Everything is in [`/code/muon-ns-steps/`](/code/muon-ns-steps/README.md), also
as one [tarball](/code/muon-ns-steps.tar.gz): the training script, the
per-run JSON (curves, timings, end-of-training momentum spectra) and the
analysis and chart scripts.

```sh
pip install torch==2.14.1 numpy --index-url https://download.pytorch.org/whl/cpu
./run_all.sh     # downloads tiny-shakespeare; ~80 min on 4 cores
```

A single run (`THREADS=1 python3 muon_ns.py --arm ns5 --lr 0.02 --seed 0`)
takes about 2 minutes and, single-threaded, reproduces the shipped 1.6804
exactly; more threads move the last digits.
