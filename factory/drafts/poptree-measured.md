---
title: "PopTree, measured: a balanced tree that wants its hot keys far apart"
description: "I implemented the new dynamically optimal weight-balanced tree. Spread 729 hot keys across the key space and it touches 38% fewer nodes than a static balanced tree; put them side by side and the saving is gone."
date: 2026-10-22
draft: false
---

Give a self-adjusting balanced tree 729 hot keys out of 59,049 and it gets a
lot cheaper, unless the 729 keys are adjacent. Then it costs about what an
ordinary balanced tree costs, as if nothing were hot. A new paper proves that
something like this has to happen. I implemented its tree to see how big the
effect is.

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
best it can do, even knowing the future, is a cost they call OPT(X), up to a
constant factor. They also build an online tree, the **PopTree**, that achieves
it. The formula has a twist. In the paper's words, "arbitrary BSTs reward
locality in key-space while weight-balanced BSTs punish it". Under OPT,
searching x costs little only when no key near x in sorted order has come up
since x was last searched.

## The question

The paper is all proofs: no code, no experiments, and constants nobody has
pinned down. Three things I wanted to know:

1. How large is the locality penalty in practice? Is it a factor of two or a
   rounding error?
2. Does PopTree beat a plain balanced tree, and a splay tree, on realistic
   access patterns, once all three are measured in the same unit?
3. What does it cost when its adaptation can't help?

The paper already says that a repeated sequential scan costs Θ(log n) per
search on any BB[α] tree. It also proves that if keys are randomly shuffled,
OPT is, in expectation, asymptotically the working-set bound, which splay
trees also achieve. So PopTree should be within a constant factor of splay on
shuffled keys. I treat those as predictions to check, not as findings.

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

To compare it with BSTs, I turn the skip list into a BST the way the paper's
Appendix B does. The one key on level 1 is the root. The keys that first
appear between two consecutive level-ℓ keys (at most four) form a group, and
each group becomes a right-leaning chain, smallest key on top. Every structure
is then charged **nodes touched per search**: the depth of the key before the
search, plus one. A checker verifies every invariant from the paper after
every search at n = 3^8, on all 12 workloads (60,000 searches each); none
failed. The resulting tree's worst w(child)/w(parent), sampled every 97
searches, was 0.101, above the 1/20 the paper guarantees after each search.

The opponents:

- **static balanced**: a perfectly balanced BST that never changes. It's the
  `Data.Map` situation for a fixed key set.
- **splay**: a bottom-up splay tree, starting from the same balanced shape.
- **offline entropy BST**: a static tree built after seeing the whole access
  sequence, with each root splitting the access frequency in half. It obeys no
  balance rule, so it shows what knowing the hot set is worth without one.

For the main grid, n = 3^10 = 59,049 keys and m = 2,000,000 searches, with
5 seeds. I measure only the second million, after everything has adapted (the
theorem needs m = Ω(n log n)). I repeated the grid at n = 3^12 with 20 million
searches (3 seeds). The hot-set workloads draw uniformly from n/81 keys (729 at
n = 3^10, 6,561 at n = 3^12), placed at random, adjacent, or evenly spaced S
apart. The whole run takes 10 minutes
on 4 CPU cores. The unit is nodes touched, not nanoseconds, because PopTree is
a skip list here, not a pointer-based tree.

## Results

Nodes touched per search, n = 59,049, mean of 5 seeds; the bracket is the
min–max where seeds differ by more than 0.1 (node-count columns only;
restructure ops are means). Restructure ops are PopTree's
skip-list promotions plus demotions per search, and they're not counted in its
column. Splay's rotations (one fewer than the nodes on its path, so about 20
per search on uniform access) aren't counted in its column either.

| workload | PopTree | splay | static balanced | offline entropy BST | PopTree restructure ops |
|---|---|---|---|---|---|
| uniform | 16.28 | 20.94 | 14.89 | 14.85 | 21.7 |
| Zipf, keys shuffled | 11.56 [11.50–11.64] | 13.48 | 14.87 [14.51–15.09] | 9.67 [9.37–9.83] | 14.5 |
| Zipf, keys adjacent | 15.63 [15.47–15.80] | 14.06 | 14.87 [14.75–14.96] | 10.11 | 21.2 |
| 729 hot keys, random spots | 10.54 | 11.76 | 14.92 | 8.61 | 12.8 |
| 729 hot keys, 81 apart | 9.27 | 11.76 | 14.91 | 8.61 | 11.4 |
| 729 hot keys, adjacent | 14.71 [13.94–15.77] | 11.76 | 14.89 | 8.61 | 18.1 |
| random walk ±8 | 15.59 [15.05–16.13] | 4.45 | 14.89 | 12.57 [11.65–13.27] | 20.3 |
| scan, ascending | 11.39 | 5.42 | 14.89 | 14.89 | 19.5 |
| scan, descending | 26.28 | 5.42 | 14.89 | 14.89 | 19.5 |

<div id="pt-workload" style="width:100%;min-height:560px" role="img" aria-label="Horizontal grouped bar chart: nodes touched per search for PopTree, splay tree, static balanced BST and offline entropy BST on eight workloads at n = 59,049."></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

*Nodes touched per search, mean of 5 seeds; whiskers show min–max over seeds.
Hover for values.*

The hot-set rows are one experiment: 729 keys, drawn the same way, with only
their spacing changing.

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

- **Spacing alone moves PopTree from 14.7 to 9.3 nodes.** With the hot keys 81
  apart, it's within 8% of the offline entropy tree (8.61), which knew the hot
  set in advance. With them adjacent, it's at the static tree's 14.9, and on
  one seed it was worse (15.77). Each tripling of the spacing saves 0.6 to
  1.7 nodes.
- **Splay doesn't care about spacing.** It's 11.76 in every hot-set row, so it
  loses to PopTree once the keys are 27 or more apart, and wins when they're
  closer. It pays for its adaptivity on uniform access: 20.9 nodes against the
  static tree's 14.9.
- **My runs are consistent with the paper's predictions, and the constants
  are visible.** With Zipf
  popularity over shuffled keys, PopTree lands within a constant of splay as
  predicted; in my runs it's ahead, 11.6 against 13.5. Random walks get
  nothing from it (15.6 against the static 14.9), while splay gets 4.5.
- **Adjacent popularity gets nothing either.** The same Zipf distribution over
  adjacent keys costs PopTree 15.6 against the static tree's 14.9. That's the
  locality penalty again, on a workload that isn't a toy.
- **When it can't help, it costs.** On uniform access it's 9% worse than the
  static tree, and it does 22 promotions and demotions per search to get there.
- **Scans are a layout story.** On every scan search, the key is at the bottom
  level of the skip list (level 11 of 11), so PopTree never adapts, as the
  theorem says. What that costs in BST nodes depends on the direction:
  the right-leaning chains make an ascending scan cheap (11.4) and a
  descending one expensive (26.3).
- **Larger n doesn't change the picture.** At n = 531,441, the ratio of
  PopTree to static is 1.10 on uniform, 0.75 on shuffled Zipf, 0.71 on the
  81-apart hot set (now 6,561 keys) and 1.04 on the adjacent one.

## What it means

The reason is visible in the balance rule. As the paper notes, two adjacent
keys can't both be shallow. Here's why. In any BST, two keys that are
adjacent in sorted order are ancestor and descendant. Say x is the ancestor.
Then x+1 is the leftmost node of x's right subtree. Under BB[α], that subtree
holds at least an α fraction of x's weight. Each step down its left spine
keeps at most a 1−α fraction of the weight, so reaching the bottom of a
subtree of size s takes at least log(s)/log(1/(1−α)) steps. If x sits near the
root, its subtree is large, so x+1 sits about log n deep. The paper's Lemma 5
generalizes this: of two keys r apart, one must be Ω(log(n/r)) deep. A balanced
tree can keep 729 keys spread over the whole space all near the top. It can't
do that for 729 neighbours, because at most a few of them can be shallow at
once. PopTree's demotion rule is that argument turned into code. The offline
entropy tree has no balance rule, so it packs adjacent hot keys into a compact
subtree, and its cost (8.61) doesn't depend on spacing.

Takeaways:

- If you want a self-adjusting tree that stays balanced, its benefit depends on
  where your hot keys are, not just how many there are. Random IDs and hashed
  keys are a good case. Keys whose hot set is a contiguous range, such as
  recent timestamps or sequential IDs, should be close to its worst, though I
  only tested synthetic versions of that.
- For scans, walks and contiguous hot ranges, a balanced tree can't adapt, by
  theorem. You need a splay tree, a finger search, or a cache in front of the
  tree.
- On access patterns it can't exploit, PopTree touches about as many nodes as
  doing nothing or more (9% more on uniform, 77% more on a descending scan) and
  does about 20 restructuring operations per search. A plain
  balanced tree is the safer default unless you know your hot keys are spread
  out.

### Where it breaks

- **The unit is nodes touched, not time.** I didn't build a pointer-based
  PopTree with rotations. The restructuring column shows that its upkeep is not
  free: about 11 to 27 skip-list operations per search across all runs.
- **The BST layout matters by a constant.** The paper's chains are one way to
  lay out each group. A median-split layout, which I also measured, costs 1–6%
  less on every non-scan workload at both sizes and flips the scans (21.4
  ascending, 16.2 descending at n = 3^10). The spacing curve, and at n = 3^10
  the splay crossover between spacings 9 and 27, hold in both. All columns are in `results/summary.json`, along with the
  layout-free skip-list level.
- **One free choice I made.** When gap repair can promote one of several
  children, the paper allows any. I promote the most recently searched one.
  Promoting the leftmost instead changes PopTree's cost by at most 0.06 nodes on
  the six workloads I checked (`ablate_fill.sh`). That bound is for the paper's
  chain layout; under the median layout the ascending scan drops from 21.4 to
  16.7.
- **Static key set, searches only.** The paper's model has no inserts or
  deletes, and n is a power of three. So is mine.
- **Synthetic workloads.** The hot-set rows are deliberately clean. Real traces
  mix adjacency and spread, and OPT is about how they mix.
- **OPT(X) has unknown constants.** I evaluated the formula as written. Across
  the 12 workloads, its rank correlation with PopTree's skip-list level (the
  quantity the paper bounds) is 0.92 at n = 3^10 and 0.87 at n = 3^12. It
  can't tell you the absolute cost.

## Reproduce it

Everything is in [`/code/poptree-measured/`](/code/poptree-measured/README.md),
also as one [tarball](/code/poptree-measured.tar.gz): a single C++ file with
the PopTree, the baselines, the OPT terms and the invariant checker, plus the
run scripts and the raw results. No dependencies beyond g++ and Python 3's
standard library.

```sh
./run.sh            # invariant check + both grids, ~10 min on 4 cores
./ablate_fill.sh    # gap-repair ablation, ~1 min
python3 analyze.py && python3 chart_data.py
```
