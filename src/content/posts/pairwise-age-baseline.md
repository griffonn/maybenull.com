---
title: "Pairwise learning, against a baseline that learns"
description: A new paper reports that training on "which face is older?" cuts small-data age error from 22.6 to 13.9 years. On the paper's kind of split, predicting one constant age scores 22.2. Plain MSE regression, same network, gets 12.0–13.7 at N=100 and beats or ties pairwise at every N I tried.
date: 2026-10-22
draft: false
---

A new paper trains a network on pairs of faces ("which one is older?") instead
of on ages. With 100 training faces it reports 13.9 years of mean absolute error,
against 22.6 for ordinary regression. On that kind of split, predicting one
age for every face scores 22.2.

## Context

[Pairwise Learning Increases Data Efficiency in Deep Neural Networks](https://arxiv.org/abs/2610.27669)
(Bain, Rashidi and Chrastil, October 2026,
[code](https://github.com/bainro/pairwise_learning)) starts from a familiar
complaint. Deep regressors trained on a few hundred examples tend to predict
roughly the training mean for everything. The paper calls this regression
collapse.

Its fix changes the supervision, not the model. Two grayscale faces are stacked
as a 2-channel image. A small ResNet ("ResNet-10", 4 residual blocks) outputs
two logits for "first is older" and "second is older", trained with
cross-entropy. One labeled set of N images yields about N² ordered pairs, so 100
labels become ~10,000 training examples.

To predict an age, the model compares the test face against every labeled
reference face. It averages P(test is older) per reference age, smooths that
curve, and picks the age where the probability is closest to 0.5 (maximum
entropy). On UTKFace, a public set of 23,708 faces labeled with age, the paper
reports MAE 13.9 vs. 22.6 for one N=100 run, a gap it finds significant across
5 paired seeds (p = .003), and 3.64 years on the full dataset.

## The question

A 22.6-year error at N=100 is a strong claim about the baseline, so that's
where I looked. The pointwise UTKFace baseline isn't in the repository. The
paper says it tried a tanh output, an unnormalized output and a linear rescale.
It doesn't say what the rescale was. Standardizing the target (subtracting
the mean age and dividing by the standard deviation) is the usual choice, and it
is itself a linear rescale, so I ran MSE both with and without it.

So, three questions:

1. What does a constant prediction score on this kind of split?
2. Does pairwise still win against a plain MSE baseline, with and without a
   standardized target, at the same model, optimizer and step budget?
3. Does the gap grow or shrink with N?

## Setup

**Data.** UTKFace, the 23,708 cropped faces, converted to grayscale with the
formula from the paper's code and resized to 64×64. That is smaller than the paper's 192×192,
to fit a CPU budget.

**Splits.** The test set is 200 faces at evenly spaced positions in the
age-sorted dataset, which is how the paper describes its N=100 test set. It's
the same for every run. The repo's dataset class draws its 200 test faces at
random instead. Over 20 such random draws, the constant below scores
21.5 ± 1.1 (19.3–23.3), so this choice doesn't change the story.

Training sets follow the repo's `PairwiseUTKFace` class: N/10 faces from each age decade (0–9 through 90+). So
the training set is age-balanced, while the test set follows the dataset, which
is mostly 20- and 30-year-olds. The paper doesn't say which split its N=100
figure used, so I also ran a natural (uniformly sampled) training set.

**Arms.** All use the repo's `AgeRanker` architecture as its forward pass
actually runs (four residual stages; a fifth is defined but never called),
AdamW at 5e-4, batch 32, random horizontal flips, and the same number of steps (a pairwise step sees 32 pairs, so it touches
twice as many images).
None use early stopping; I report the last checkpoint.

- `pair`: the paper's method, with the training images as references.
- `mse`: one image in, one number out, MSE on the standardized age.
- `l1`: the same with L1 loss, which targets the median instead of the mean.
- `mse_raw`: MSE on raw age in years, no standardization.
- Two model-free baselines: the training-set mean, and ridge regression on raw
  pixels.

The standardized target is two lines:

```python
mu, sd = ages[train].mean(), ages[train].std()
loss = F.mse_loss(model(x).squeeze(1), (y - mu) / sd)   # predict: out * sd + mu
```

The pairwise label and decode, condensed from the repo:

```python
lab = (y[i] <= y[j]).long()                  # 0 = first face is older
p_older = model(torch.cat([x_test, x_ref], 1)).softmax(1)[:, 0]
votes = [p_older[ref_age == a].mean() for a in unique_ages]
v = gaussian_filter1d(votes, 1.0)
pred = unique_ages[argmax(-(v*log v + (1-v)*log(1-v)))]
```

Each run is single-threaded on a 4-core Xeon (2.1 GHz), 4 runs at a time.
3,000 steps take 4 minutes for `mse` and 6–29 minutes for `pair`, depending on
N. Pairwise inference costs one forward pass per reference image, so at N=1000
that is 1,000 passes per test face.

## Results

Test MAE in years, mean ± standard deviation over seeds (5 seeds; 3 for the
N=1000 network runs and the 10k-step rows). Decade-balanced training sets.

| | N=100 | N=100, 10k steps | N=300 | N=1000 |
|---|---|---|---|---|
| constant: training-set mean | 22.2 ± 0.3 | | 22.2 ± 0.1 | 22.1 ± 0.0 |
| MSE on raw age | 13.7 ± 1.1 | | | |
| ridge on raw pixels | 17.2 ± 1.4 | | 15.5 ± 0.5 | 14.4 ± 0.3 |
| **pairwise (paper)** | 13.7 ± 1.3 | 13.0 ± 0.3 | 12.7 ± 1.2 | 10.3 ± 0.1 |
| **MSE, standardized** | 12.7 ± 0.6 | 12.0 ± 0.6 | 10.7 ± 0.7 | 8.5 ± 0.3 |
| L1, standardized | 12.0 ± 0.7 | | | |

The paper's numbers at N=100, for reference: pairwise 13.9, MSE 22.6.

<div id="pw-mae" style="width:100%;min-height:460px" role="img" aria-label="Test MAE in years versus number of labeled training images (100, 300, 1000) for pairwise, standardized MSE, L1, raw-age MSE, ridge regression on pixels and a constant training-mean predictor, with the paper's reported MSE baseline of 22.6 marked."></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function () {
  var C = {pair: '#d1495b', mse: '#2e86ab', l1: '#66a182', mse_raw: '#7b5ea7', ridge_pixels: '#edae49', const_train_mean: '#999999'};
  var L = {pair: 'pairwise CE + entropy decode (paper)', mse: 'pointwise MSE, standardized target',
    l1: 'pointwise L1, standardized target', mse_raw: 'pointwise MSE, raw age', ridge_pixels: 'ridge regression on raw pixels',
    const_train_mean: 'constant: training-set mean'};
  function theme() {
    var cs = getComputedStyle(document.documentElement);
    return {fg: cs.getPropertyValue('--fg').trim(), muted: cs.getPropertyValue('--muted').trim(), grid: cs.getPropertyValue('--line').trim()};
  }
  function axis(t, extra) { return Object.assign({gridcolor: t.grid, linecolor: t.muted, zerolinecolor: t.muted, tickfont: {color: t.muted}}, extra); }
  function base(t, narrow, h) {
    return {height: h, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', hovermode: 'closest',
      font: {family: 'Inter, system-ui, sans-serif', color: t.fg, size: 12}, annotations: [], shapes: [],
      legend: {orientation: 'h', x: 0, y: narrow ? -0.16 : -0.2, yanchor: 'top', font: {color: t.fg}}};
  }
  function drawMae(d) {
    var el = document.getElementById('pw-mae'), t = theme(), narrow = el.clientWidth < 560, traces = [];
    var lay = base(t, narrow, narrow ? 620 : 480);
    lay.margin = {l: 52, r: 10, t: 10, b: narrow ? 210 : 140};
    lay.xaxis = axis(t, {type: 'log', tickvals: [100, 300, 1000], ticktext: ['100', '300', '1000'],
      title: {text: 'labeled training images (decade-balanced)', font: {color: t.muted}, standoff: 6}});
    lay.yaxis = axis(t, {title: {text: 'test MAE, years (lower is better)', font: {color: t.muted}}, range: [6, 25]});
    lay.shapes.push({type: 'line', xref: 'x domain', x0: 0, x1: 1, yref: 'y', y0: d.paper_mse_baseline, y1: d.paper_mse_baseline,
      line: {color: t.muted, width: 1, dash: 'dot'}});
    lay.annotations.push({text: "paper's reported MSE baseline at N=100 (22.6)", xref: 'x domain', x: 0.01, xanchor: 'left',
      yref: 'y', y: d.paper_mse_baseline, yanchor: 'bottom', showarrow: false, font: {color: t.muted, size: 11}});
    ['const_train_mean', 'ridge_pixels', 'pair', 'mse', 'l1', 'mse_raw'].forEach(function (a) {
      var r = d.arms[a]; if (!r) return;
      var dashed = a === 'const_train_mean' || a === 'ridge_pixels';
      if (r.n.length > 1 && !dashed) traces.push({x: r.n.concat(r.n.slice().reverse()), y: r.max.concat(r.min.slice().reverse()),
        mode: 'lines', fill: 'toself', fillcolor: C[a] + '26', line: {width: 0}, hoverinfo: 'skip', showlegend: false, legendgroup: a});
      traces.push({x: r.n, y: r.mean, name: L[a], legendgroup: a, mode: 'lines+markers',
        line: {color: C[a], width: 2.5, dash: dashed ? 'dash' : 'solid'}, marker: {size: 7},
        customdata: r.min.map(function (v, j) { return [v, r.max[j], r.seeds[j]]; }),
        hovertemplate: dashed ? 'N = %{x}<br>MAE %{y:.1f} years<extra>' + L[a] + '</extra>'
          : 'N = %{x}<br>MAE %{y:.1f} years (mean of %{customdata[2]})<br>range %{customdata[0]:.1f}–%{customdata[1]:.1f}<extra>' + L[a] + '</extra>'});
    });
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function drawScatter(d) {
    var el = document.getElementById('pw-scatter'), t = theme(), narrow = el.clientWidth < 560, traces = [];
    var lay = base(t, narrow, narrow ? 720 : 420);
    lay.margin = {l: 52, r: 10, t: 28, b: 50};
    lay.showlegend = false;
    lay.grid = narrow ? {rows: 2, columns: 1, pattern: 'independent', ygap: 0.18} : {rows: 1, columns: 2, pattern: 'independent', xgap: 0.08};
    [['mse', 'pointwise MSE'], ['pair', 'pairwise']].forEach(function (p, i) {
      var s = i === 0 ? '' : '2', r = d[p[0]];
      lay['xaxis' + s] = axis(t, {range: [0, 100], title: {text: 'true age', font: {color: t.muted}, standoff: 4}});
      lay['yaxis' + s] = axis(t, {range: [0, 100], title: {text: 'predicted age', font: {color: t.muted}}});
      lay.annotations.push({text: p[1] + ': MAE ' + r.mae.toFixed(1), xref: 'x' + s + ' domain', yref: 'y' + s + ' domain',
        x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: {color: t.fg, size: 12}});
      lay.shapes.push({type: 'line', xref: 'x' + s, yref: 'y' + s, x0: 0, y0: 0, x1: 100, y1: 100, line: {color: t.muted, width: 1}});
      lay.shapes.push({type: 'line', xref: 'x' + s + ' domain', x0: 0, x1: 1, yref: 'y' + s, y0: r.train_mean, y1: r.train_mean,
        line: {color: t.muted, width: 1, dash: 'dot'}});
      traces.push({x: r.true, y: r.pred, xaxis: 'x' + s, yaxis: 'y' + s, mode: 'markers', name: p[1],
        marker: {color: C[p[0]], size: 6, opacity: 0.7},
        hovertemplate: 'true %{x:.0f}, predicted %{y:.1f}<extra>' + p[1] + '</extra>'});
    });
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function load(url) { return fetch(url).then(function (r) { return r.json(); }); }
  function go() {
    Promise.all([load('/lab/pairwise-age-baseline/mae-vs-n.json'), load('/lab/pairwise-age-baseline/scatter-n100.json')]).then(function (v) {
      function all() { drawMae(v[0]); drawScatter(v[1]); }
      all();
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', all);
      var el = document.getElementById('pw-mae'), narrow = el.clientWidth < 560;
      window.addEventListener('resize', function () { if ((el.clientWidth < 560) !== narrow) { narrow = !narrow; all(); } });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
</script>

*Mean over seeds with the min–max range shaded; dashed lines are the
non-neural baselines. Hover for values.*

Plainly:

- **A constant matches the 22.6.** A decade-balanced training set has a mean
  age near 50, and the test set's median is 29. Predicting the training mean
  (about 49) for every face gives 22.2 on average. A baseline at 22.6 does
  no better than predicting one age for everyone.
- **My pairwise arm reproduces the paper.** 13.7 at 3k steps and 13.0 at
  10k steps, against the paper's 13.9 (a single run). The change of input resolution didn't
  break it.
- **Standardized MSE matches or beats pairwise at every N.** At N=100 the gap is within
  noise at 3k steps: pairwise is ahead on 2 of 5 paired seeds, by 0.1–0.2 years.
  Given 10k steps, MSE is ahead on all 3 seeds, by 0.2–1.5 years. At N=300 and
  N=1000, MSE is ahead on every seed, by 0.8–3.0 years. The gap doesn't close as
  N grows.
- **Neither MSE arm collapses.** Even on raw ages, with no standardization,
  MSE reaches 13.7 ± 1.1 at N=100, a tie with pairwise's 13.7 ± 1.3. On the
  balanced splits, the standardized model's predictions keep 73–94% of the true
  ages' spread (85% for raw ages). Pairwise keeps 99–111%. It does spread its
  predictions more, and it's still no more accurate.
- **With a natural training set**, the constant scores 16.0, MSE 11.1 ± 0.8 and
  pairwise 12.4 ± 1.5 (3 seeds). Same ordering.

<div id="pw-scatter" style="width:100%;min-height:420px" role="img" aria-label="Predicted versus true age for 200 test faces at N=100 (10k steps, seed 0), pointwise MSE and pairwise side by side, with the identity line and the training-set mean marked."></div>

*Solid line: perfect prediction. Dotted line: the training-set mean.*

The scatter (10k-step runs, seed 0) shows the difference in kind. MSE shrinks toward the middle: it
over-predicts infants and under-predicts the very old. Pairwise keeps the full
range, but it's noisier in the 20–60 band where most faces are. It can also
only answer with ages that exist in the reference set.

One more number, about protocol. The paper says it early-stops on its test
set (its script tracks test-set pairwise accuracy). The equivalent here (best of
the evaluated checkpoints by test MAE) lowers the error by 0.1–0.7 years for
most arms. Raw-target MSE gains the most, 2.7 years (13.7 to 10.9), because its
best checkpoints come early and it drifts afterwards. Picking on the test set
flatters exactly the runs that are least stable, and it isn't a held-out number.

## What it means

The pairwise idea is fine, and it works. It just isn't better than regression
done the standard way, at least not here. Turning one label into many pairs
sounds like more supervision. My read is that N labels still carry N ages'
worth of information however you slice them, and a ranking loss throws away
the scale. The decode has
to put the scale back by comparing against references.

Most of the headline gap comes from the baseline. I tried the obvious ways to
make plain regression fail (raw targets, standardized targets, L1, short and
long training) and none of them came near 22.6. The worst any MSE run did at
N=100 was 15.5. The paper's own Fig. 1A shows its baseline's predictions
bunched near the training mean. Whatever that baseline did, it scored like a
model that learned nothing, and a constant is enough to reproduce its number.

Takeaways:

- If your small-data regressor predicts the mean, check the baseline before you
  change the loss: standardize the target, check the learning rate, and confirm
  it beats a constant.
- Always report a constant baseline. When the training and test distributions
  differ, as they do with decade-balanced sampling, that constant can be far
  worse than the test set's own median (15.2 here). A 22-year baseline should
  look suspicious.
- Pairwise training plus decode costs N forward passes per prediction (8N with
  the paper's test-time augmentation; its binary-search decode trims this at
  full scale). It buys spread, not accuracy.

**Where this breaks.** This is 64×64 input, not 192×192, and has no 8×
test-time augmentation for any arm. My N=100 pairwise result matches the
paper's, but higher resolution could favor one arm. I didn't tune either arm
beyond the repo's optimizer settings, and I used a constant learning rate where
the paper's script uses ReduceLROnPlateau; tuning would likely help both. I can't
see the paper's actual baseline code, so "it learned nothing" is an inference
from its score, not from its code. I didn't test the paper's other claims: the
3.64 full-dataset result (which uses the test images as references) or the
STL-10 classification gains.

## Reproduce it

Everything is in
[`/code/pairwise-age-baseline/`](/code/pairwise-age-baseline/README.md), also as
one [tarball](/code/pairwise-age-baseline.tar.gz): the training script, the
baselines, the per-run JSON results (learning curves and per-image predictions)
and the plotting scripts. UTKFace isn't included; the README says where to get
it.

```sh
python3 prep.py /path/to/UTKFace data/utk64.npz   # 30 s
python3 baselines.py                              # 10 s
python3 const_random_test.py                      # 1 s
./run.sh && ./run_long.sh && ./run_raw.sh         # ~2 h on 4 cores
python3 analyze.py && python3 plot.py
```
