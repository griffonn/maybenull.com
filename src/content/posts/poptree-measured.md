---
title: "PopTree, measured: a balanced tree that wants its hot keys far apart"
description: "I implemented the new dynamically optimal weight-balanced tree. Spread 729 hot keys across the key space and it touches 40% fewer nodes than a static balanced tree; put them side by side and the saving is gone."
date: 2026-10-22
draft: false
---

Give a self-adjusting balanced tree 729 hot keys out of 59,049 and it gets a
lot cheaper, unless the 729 keys are adjacent. Then it costs about what an
ordinary balanced tree costs, as if nothing were hot. A new paper proves this
has to happen. I implemented its tree to see how big the effect is.

## Context

A *weight-balanced* tree, or BB[α] tree, is a binary search tree (BST) in which
every child's subtree holds at least a fraction α of its parent's. It was
introduced in 1972, and it's the tree behind Haskell's `Data.Map`, which the
docs describe as
["size balanced binary trees (or trees of bounded balance)"](https://hackage-content.haskell.org/package/containers-0.8.1/docs/Data-Map-Strict.html).
Balance keeps every search logarithmic, but `Data.Map` doesn't adapt: a hot key
stays as deep as it was.

The other classic option is the splay tree. It rotates every key it finds to
the root, so a key that's searched often stays near the top. Nobody has proved
that splay trees are *dynamically optimal*, meaning within a constant factor of
the best possible rotating BST on every access sequence. That question has been
open since 1985.

[Iacono and Pareek](https://arxiv.org/abs/2610.29574) (October 2026) settle the
same question for balanced trees. If a tree must stay BB[α] at all times, the
best it can do, even knowing the future, is a cost they call OPT(X). They also
build an online tree, the **PopTree**, that achieves it up to a constant. The
formula has a twist. In the paper's words, arbitrary BSTs reward locality in
key space, while weight-balanced BSTs "punish it". A search is cheap only if
nothing *close to it in key order* was searched since that key's last search.

## The question

The paper is all proofs: no code, no experiments, and constants nobody has
pinned down. Three things I wanted to know:

1. How large is the locality penalty in practice? Is it a factor of two or a
   rounding error?
2. Does PopTree beat a plain balanced tree, and a splay tree, on realistic
   access patterns, once all three are measured in the same unit?
3. What does it cost when its adaptation can't help?

The paper already says that a sequential scan costs Θ(log n) per search on any
BB[α] tree. It also proves that if keys are randomly shuffled, OPT equals the
working-set bound, which splay trees also achieve. I treat those two as
predictions to check, not as findings.

## Setup

I implemented the PopTree in C++ from Section 3.1 of the paper. It's described
as a skip list over n = 3^L keys. At level ℓ the key space is cut into 3^ℓ
equal buckets, and each bucket holds at most one key. A search *pops* the found
key up to the highest level where its bucket isn't at the edge of the key
space. It demotes any key sitting in the same bucket or in either neighbour,
then patches any run of three empty buckets by promoting a key from the level
below:

```cpp
// clearing demotions, levels [t, li): nobody may sit in x's bucket or next to it
for (int l = t; l < li; l++)
    for (ll b = bucket(l, x) - 1; b <= bucket(l, x) + 1; b++)
        if (int y = occ[l][b]; y && y != x)
            while (top[y] <= L && llabs(bucket(top[y], y) - bucket(top[y], x)) <= 1)
                demote_top(y);
// promotion: x now holds its bucket at every level from li-1 up to t
for (int l = li - 1; l >= t; l--) occ[l][bucket(l, x)] = x;
top[x] = t;
// gap repair, bottom-up: three empty buckets in a row get a key from a child bucket
```

That demotion rule is where locality gets punished: popping x pushes its key
neighbours down.

To compare it with BSTs, I convert the skip list to a BST the usual way. The
one key on level 1 is the root, and the keys that first appear between two
consecutive level-ℓ keys form a small balanced subtree. Every structure is
then charged **nodes touched per search**: the depth of the key before the
search, plus one. A checker verifies every invariant from the paper after every
search at n = 3^8, on all 11 workloads (60,000 searches each); none failed. The
converted tree's worst w(child)/w(parent), sampled every 97 searches, was
0.128. (The paper guarantees 1/20 after each search for its own layout.)

The opponents:

- **static balanced**: a perfectly balanced BST that never changes. It's the
  `Data.Map` situation for a fixed key set.
- **splay**: a bottom-up splay tree, starting from the same balanced shape.
- **offline entropy BST**: a static tree built after seeing the whole access
  sequence, with each root splitting the access frequency in half. It obeys no
  balance rule, so it shows what knowing the hot set is worth without one.

For the main grid, n = 3^10 = 59,049 keys and m = 2,000,000 searches, with
5 seeds. I measure only the second million, after everything has adapted (the
theorem needs m ≥ n log n). I repeated the grid at n = 3^12 with 20 million
searches (3 seeds). The hot-set workloads draw uniformly from 729 keys, placed
at random, adjacent, or evenly spaced S apart. The whole run takes 9 minutes
on 4 CPU cores. The unit is nodes touched, not nanoseconds, because PopTree is
a skip list here, not a pointer-based tree.

## Results

Nodes touched per search, n = 59,049, mean of 5 seeds; the bracket is the
min–max where seeds differ by more than 0.1. Restructure ops are PopTree's
skip-list promotions plus demotions per search, and they're not counted in its
column.

| workload | PopTree | splay | static balanced | offline entropy BST | PopTree restructure ops |
|---|---|---|---|---|---|
| uniform | 15.54 | 20.94 | 14.89 | 14.85 | 21.7 |
| Zipf, shuffled keys | 11.02 [10.94–11.09] | 13.48 | 14.87 [14.51–15.09] | 9.67 [9.37–9.83] | 14.5 |
| Zipf, adjacent keys | 15.08 [14.98–15.15] | 14.06 | 14.87 [14.75–14.96] | 10.11 | 21.2 |
| 729 hot keys, random spots | 10.05 | 11.76 | 14.92 | 8.61 | 12.8 |
| 729 hot keys, 81 apart | 8.92 | 11.76 | 14.91 | 8.61 | 11.4 |
| 729 hot keys, adjacent | 14.49 [13.66–15.60] | 11.76 | 14.89 | 8.61 | 18.1 |
| random walk ±8 | 15.27 [15.10–15.73] | 4.45 | 14.89 | 12.57 [11.65–13.27] | 20.3 |
| sequential scan | 21.45 | 5.42 | 14.89 | 14.89 | 19.5 |

<div id="pt-workload" style="width:100%;min-height:520px" role="img" aria-label="Horizontal grouped bar chart: nodes touched per search for PopTree, splay tree, static balanced BST and offline entropy BST on seven workloads at n = 59,049."></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

*Nodes touched per search, mean of 5 seeds; whiskers show min–max over seeds.
Hover for values.*

The hot-set rows are one experiment: the same 729 keys, drawn the same way,
with only the spacing changing.

<div id="pt-stride" style="width:100%;min-height:440px" role="img" aria-label="Line chart: nodes touched per search versus spacing between 729 hot keys (1, 3, 9, 27, 81) for PopTree, splay, static balanced BST and offline entropy BST at n = 59,049."></div>
<script>
(function () {
  var ARMS = ['poptree', 'splay', 'static', 'entropy_static'];
  var C = {poptree: '#0072B2', splay: '#E69F00', static: '#009E73', entropy_static: '#CC79A7'};
  var NAME = {poptree: 'PopTree', splay: 'splay tree', static: 'static balanced BST', entropy_static: 'offline entropy BST'};
  function theme() {
    var cs = getComputedStyle(document.documentElement);
    return {fg: cs.getPropertyValue('--fg').trim(), muted: cs.getPropertyValue('--muted').trim(), grid: cs.getPropertyValue('--line').trim()};
  }
  function axis(t, extra) { return Object.assign({gridcolor: t.grid, linecolor: t.muted, zerolinecolor: t.muted, tickfont: {color: t.muted}}, extra); }
  function base(t, narrow, h) {
    return {height: h, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', hovermode: 'closest',
      font: {family: 'Inter, system-ui, sans-serif', color: t.fg, size: 12},
      legend: {orientation: 'h', x: 0, y: narrow ? -0.12 : -0.16, yanchor: 'top', font: {color: t.fg}}};
  }
  function err(r) {
    return {type: 'data', symmetric: false, array: r.max.map(function (v, i) { return v - r.mean[i]; }),
      arrayminus: r.min.map(function (v, i) { return r.mean[i] - v; }), thickness: 1.2, width: 3};
  }
  function drawWorkload(d) {
    var el = document.getElementById('pt-workload'), t = theme(), narrow = el.clientWidth < 560;
    var lay = base(t, narrow, narrow ? 700 : 560);
    lay.margin = {l: narrow ? 118 : 190, r: 10, t: 40, b: narrow ? 130 : 110};
    lay.title = {text: narrow ? 'Cost by workload, n = 59,049' : 'Cost per search by workload, n = 59,049 keys', x: 0, xref: 'paper', font: {color: t.fg, size: 14}};
    lay.barmode = 'group'; lay.bargap = 0.25; lay.bargroupgap = 0.05;
    var labels = d.labels.map(function (l) { return narrow ? l.replace(', ', ',<br>') : l; });
    lay.yaxis = axis(t, {autorange: 'reversed', automargin: true, tickfont: {color: t.fg, size: narrow ? 11 : 12}});
    lay.xaxis = axis(t, {rangemode: 'tozero', title: {text: narrow ? 'nodes per search (lower = better)' : 'nodes touched per search (lower is better)', font: {color: t.muted}, standoff: 6}});
    var traces = ARMS.map(function (a) {
      var r = d.arms[a];
      return {type: 'bar', orientation: 'h', name: NAME[a], y: labels, x: r.mean, error_x: Object.assign(err(r), {color: t.muted}),
        marker: {color: C[a]}, customdata: r.min.map(function (v, i) { return [v, r.max[i]]; }),
        hovertemplate: '%{x:.2f} nodes per search<br>seeds: %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra>' + NAME[a] + '</extra>'};
    });
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function drawStride(d) {
    var el = document.getElementById('pt-stride'), t = theme(), narrow = el.clientWidth < 560;
    var lay = base(t, narrow, narrow ? 540 : 460);
    lay.margin = {l: 52, r: 10, t: 40, b: narrow ? 150 : 120};
    lay.title = {text: narrow ? '729 hot keys: spacing vs cost' : '729 hot keys out of 59,049: cost vs spacing', x: 0, xref: 'paper', font: {color: t.fg, size: 14}};
    lay.xaxis = axis(t, {type: 'log', tickvals: d.spacing, ticktext: d.spacing.map(String),
      title: {text: narrow ? 'hot-key spacing (1 = adjacent)' : 'spacing between the 729 hot keys (1 = adjacent)', font: {color: t.muted}, standoff: 6}});
    lay.yaxis = axis(t, {range: [0, 17], title: {text: 'nodes touched per search', font: {color: t.muted}}});
    var traces = ARMS.map(function (a) {
      var r = d.arms[a];
      return {x: d.spacing, y: r.mean, name: NAME[a], mode: 'lines+markers', error_y: Object.assign(err(r), {color: C[a]}),
        line: {color: C[a], width: 2.5, dash: a === 'entropy_static' ? 'dash' : 'solid'}, marker: {size: 7},
        customdata: r.min.map(function (v, i) { return [v, r.max[i]]; }),
        hovertemplate: 'spacing %{x}<br>%{y:.2f} nodes per search<br>seeds: %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra>' + NAME[a] + '</extra>'};
    });
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function load(url) { return fetch(url).then(function (r) { return r.json(); }); }
  function go() {
    Promise.all([load('/lab/poptree-measured/by-workload.json'), load('/lab/poptree-measured/stride.json')]).then(function (v) {
      function all() { drawWorkload(v[0]); drawStride(v[1]); }
      all();
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', all);
      var el = document.getElementById('pt-workload'), narrow = el.clientWidth < 560;
      window.addEventListener('resize', function () { if ((el.clientWidth < 560) !== narrow) { narrow = !narrow; all(); } });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
</script>

*Spacing 1 means the 729 hot keys are adjacent; 81 spreads them across the
whole key space. Mean of 5 seeds, whiskers min–max.*

Plainly:

- **Spacing alone moves PopTree from 14.5 to 8.9 nodes.** With the hot keys 81
  apart, it's within 4% of the offline entropy tree (8.61), which knew the hot
  set in advance. With them adjacent, it's at the static tree's 14.9, and on
  one seed it was worse (15.60). Each tripling of the spacing saves 1.0 to
  1.7 nodes.
- **Splay doesn't care about spacing.** It's 11.76 in every hot-set row, so it
  loses to PopTree once the keys are 27 or more apart, and wins when they're
  closer. It pays for its adaptivity on uniform access: 20.9 nodes against the
  static tree's 14.9.
- **The paper's two predictions hold.** Sequential and random-walk access get
  nothing from PopTree, while splay gets 5.4 and 4.5 nodes. With Zipf
  popularity over shuffled keys, PopTree beats splay (11.0 vs 13.5). Over
  adjacent keys, the same Zipf popularity gets PopTree nothing (15.1 vs 14.9
  static).
- **When it can't help, it costs.** On uniform access it's 4% worse than the
  static tree, and it does 22 promotions and demotions per search to get
  there. On a sequential scan it touches 44% more nodes than the static tree.
  That number depends on a detail the paper leaves open (see the limits
  below), but it never got below the static tree's 14.89.
- **Larger n doesn't change the picture.** At n = 531,441, the ratio of
  PopTree to static is 1.05 on uniform, 0.72 on shuffled Zipf, 0.68 on the
  spread hot set, 1.00 on the adjacent one and 1.45 on the scan.

## What it means

The reason is visible in the balance rule. In any BST, two keys that are
adjacent in sorted order are ancestor and descendant. Say x is the ancestor.
Then x+1 is the leftmost node of x's right subtree. Under BB[α], that subtree
holds at least an α fraction of x's weight. Each step down its left spine
keeps at most a 1−α fraction of the weight, so reaching the bottom of a
subtree of size s takes at least log(s)/log(1/(1−α)) steps. If x sits near the
root, its subtree is large, so x+1 sits about log n deep. The paper's Lemma 5
generalizes this: of two keys r apart, one must be Ω(log(n/r)) deep. A balanced tree can
keep 729 keys spread over the whole space all near the top. It can't do that
for 729 neighbours, because at most a few of them can be shallow at once.
PopTree's demotion rule is that argument turned into code. The offline entropy
tree has no balance rule, so it packs adjacent hot keys into a compact
subtree, and its cost (8.61) doesn't depend on spacing.

Takeaways:

- If you want a self-adjusting tree that stays balanced, its benefit depends on
  where your hot keys are, not just how many there are. Random IDs and hashed
  keys are its best case. Time-ordered keys, which make the hottest keys the
  newest and adjacent, are its worst.
- For scans, walks and recent-timestamp keys, a balanced tree can't adapt, by
  theorem. You need a splay tree, a finger search, or a cache in front of the
  tree.
- On access patterns it can't exploit, PopTree touches more nodes than doing
  nothing and does 20-odd restructuring operations per search. A plain balanced
  tree is the safer default unless you know your hot keys are spread out.

### Where it breaks

- **The unit is nodes touched, not time.** I didn't build a pointer-based
  PopTree with rotations. The restructuring column shows that its upkeep is not
  free: about 11 to 27 skip-list operations per search across all runs.
- **My BST is my conversion.** I use the standard skip list → B-tree → BST
  layout with median splits inside each group. The paper's appendix may lay
  groups out differently, which would shift PopTree's column by a constant
  factor. The ratios between workloads are what I'd trust most.
- **One free choice matters on scans.** When gap repair has a choice of
  children to promote, the paper allows any. I promote the most recently
  searched one. Promoting the leftmost child instead gives the same results on
  every workload I tried to within 0.03 nodes, except the sequential scan. On
  the scan it drops from 21.45 to 16.67 nodes, still 12% above the static tree
  (seed 1 at n = 3^10; the same direction at n = 3^8).
- **Static key set, searches only.** The paper's model has no inserts or
  deletes, and n is a power of three. So is mine.
- **Synthetic workloads.** The hot-set rows are deliberately clean. Real traces
  mix adjacency and spread, and OPT is about how they mix.
- **OPT(X) has unknown constants.** I evaluated the formula as written. Across
  the 11 workloads its rank correlation with PopTree's cost is 0.91 at
  n = 3^10 and 0.89 at n = 3^12 (values in `results/summary.json`), but it
  can't tell you the absolute cost.

## Reproduce it

Everything is in [`/code/poptree-measured/`](/code/poptree-measured/README.md),
also as one [tarball](/code/poptree-measured.tar.gz): a single C++ file with
the PopTree, the baselines, the OPT terms and the invariant checker, plus the
run scripts and the raw results. No dependencies beyond g++ and Python 3's
standard library.

```sh
./run.sh            # invariant check + both grids, ~9 min on 4 cores
./ablate_fill.sh    # gap-repair ablation, ~1 min
python3 analyze.py && python3 chart_data.py
```
