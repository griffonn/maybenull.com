---
title: "CAFS, minus the counting"
description: A new low-cardinality integer sort reports 1.7–3.1× over pdqsort. For 64-bit keys a plain hash-map count matches it from 16 to 16k distinct values, and on skewed data with many distinct values CAFS is up to 2.2× slower than that count.
date: 2026-10-21
draft: false
---

If an array of ten million integers holds only a thousand distinct values,
you don't need to sort it. You count it. A new paper sorts such arrays
2–3× faster than pdqsort, and it compares itself only against sorts that
don't count. So I gave it something that counts.

## Context

[CAFS](https://arxiv.org/abs/2605.25040) ("Cache-Aware Frequency Sort", v2
posted 19 October, [code](https://github.com/kexibq-official/cafs-lib)) targets
integer columns where the number of distinct values K is far smaller than the
length N: status codes, country ids, enum columns. These get sorted constantly in
analytics engines. A comparison sort pays O(N log N) whatever K is. A radix sort
pays one pass per key byte.

CAFS is a hash-count sort. It walks the array once, counts every distinct value
in a hash table, sorts the K (value, count) pairs, and writes each value back
`count` times. Around that core, the paper adds:

- **The bucket.** The table is an array of 64-byte buckets, one cache line each,
  holding 4 `uint64` keys (or 8 `int32` keys) and their counts. One AVX2 compare
  checks a whole bucket. When a bucket is full the value goes to a spill list,
  which is sorted with `std::sort` at the end.
- **The sampler.** Before allocating, CAFS reads 1,024 evenly spaced elements and
  estimates K with the Chao1 estimator. Chao1 counts how many sampled values
  appeared once or twice and extrapolates the unseen ones. The table gets about
  8 slots per estimated distinct value, and it never grows.
- **Run merging.** Equal neighbors are counted as one run before they reach the
  table.
- **Two exits.** If the sample shows at most 8 distinct values, CAFS skips the
  table and counts with 8 unrolled, branch-free compares. If the estimate says K is
  more than N/2, it hands the array to a regular sort.

The paper reports 1.7–3.1× over pdqsort and 1.2–2.3× over Google's vqsort when
K ≪ N. The benchmark covered 592,770 measurement rows on a Core i5-12400F with
`uint64` keys, against five general-purpose sorts.

## The question

None of those five baselines count. The paper's future-work section promises
a `std::unordered_map` hash-count baseline and says informal runs show CAFS
winning "by orders of magnitude". v2 doesn't include it.

So counting was never measured on its own. Three things I wanted to know:

1. How close does an ordinary hash-map count get to CAFS?
2. What does the AVX2 bucket add? That is an ablation: same algorithm, different
   table.
3. What happens on skewed data? The paper's grid is uniform only, and its
   conclusion lists skewed inputs as an open question.

## Setup

The counting contenders share one emit step:

```cpp
template <class Pairs>
static void emit(Vec& v, Pairs& kv) {          // kv: (value, count)
    std::sort(kv.begin(), kv.end());
    auto out = v.begin();
    for (auto& [k, c] : kv) out = std::fill_n(out, c, k);
}

static void count_ankerl(Vec& v) {
    ankerl::unordered_dense::map<Key, u64> m;  // a popular open-source hash map
    for (Key x : v) ++m[x];
    auto kv = std::move(m).extract();
    emit(v, kv);
}
```

CAFS does the same thing, except it radix-sorts the pairs once there are 256 or
more of them. At large K that favors CAFS.

The contenders:

- **CAFS** from the authors' repo, with pdqsort as its fallback.
- **pdqsort**, **ska_sort** (a radix sort) and `std::sort`.
- **Three counting sorts:** `std::unordered_map`,
  [`ankerl::unordered_dense`](https://github.com/martinus/unordered_dense), and a
  30-line linear-probing table of my own.
- **The ablation, `cafs_hashmap`.** This is CAFS's own code with the bucket table
  (and its spill list) replaced by `ankerl::unordered_dense`. The sampler, the
  ≤8-key path, run merging, the fallback rule and the radix pass over pairs are
  all unchanged.

The inputs:

- **Key types:** `uint64` (what the paper benchmarks) and `int32` (where a CAFS
  bucket holds 8 keys instead of 4).
- **Sizes:** N = 1M and 10M, with K from 2 to 1M.
- **Distributions:** uniform; Zipf with s = 1, where the most common value is
  about K times as frequent as the rarest; and clustered runs (uniform values in
  runs averaging 16 equal elements).

Every output is checked: the first repetition of each cell against `std::sort`,
later ones for sortedness, and every ablation run against `std::sort`. The
machine is a 4-vCPU KVM guest on a 2.1 GHz Xeon with AVX-512, 2 MiB of L2 per core
and a 260 MiB shared L3. I used g++ 13.3 with `-O3 -march=x86-64-v3`. Each
benchmark ran pinned to one core, 5 repetitions per cell, and I report medians.
I couldn't pin the CPU frequency, and the same cell can move 20–30% between
separate runs. Treat gaps under about 1.2× as ties. All numbers below are at
N = 10M unless noted.

## Results

<div id="cafs-ns" style="width:100%;min-height:460px" role="img" aria-label="Nanoseconds per element versus distinct values K, uniform keys, N = 10M, for CAFS, a hash-map count, pdqsort and ska_sort, in two panels: uint64 and int32 keys."></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function () {
  var NAME = {cafs: 'CAFS', count_ankerl: 'hash-map count (ankerl)', pdqsort: 'pdqsort', ska_sort: 'ska_sort (radix)'};
  var COL = {cafs: '#2a78d6', count_ankerl: '#eb6834', pdqsort: '#1baf7a', ska_sort: '#eda100',
             uniform: '#6250d6', zipf: '#e87ba4', runs: '#1baf7a'};
  var DNAME = {uniform: 'uniform', zipf: 'zipf', runs: 'clustered runs'};
  function theme() {
    var cs = getComputedStyle(document.documentElement);
    return {fg: cs.getPropertyValue('--fg').trim(), muted: cs.getPropertyValue('--muted').trim(),
            grid: cs.getPropertyValue('--line').trim()};
  }
  function base(t, el, nPanels, ytitle, ytype) {
    var narrow = el.clientWidth < 560;
    var lay = {height: narrow ? 760 : 460, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: {family: 'Inter, system-ui, sans-serif', color: t.fg, size: 12},
      margin: {l: 56, r: 10, t: 28, b: narrow ? 150 : 110}, hovermode: 'closest', annotations: [],
      grid: narrow ? {rows: nPanels, columns: 1, pattern: 'independent', ygap: 0.28} : {rows: 1, columns: nPanels, pattern: 'independent', xgap: 0.08},
      legend: {orientation: 'h', x: 0, y: narrow ? -0.12 : -0.22, yanchor: 'top', font: {color: t.fg}}};
    for (var i = 1; i <= nPanels; i++) {
      var s = i === 1 ? '' : String(i);
      var ax = {gridcolor: t.grid, linecolor: t.muted, zerolinecolor: t.muted, tickfont: {color: t.muted}};
      lay['xaxis' + s] = Object.assign({type: 'log', title: {text: 'distinct values K', font: {color: t.muted}, standoff: 6},
        dtick: 1}, ax);
      lay['yaxis' + s] = Object.assign({type: ytype || 'linear', matches: i === 1 ? undefined : 'y', title: (i === 1 || narrow) ? {text: ytitle, font: {color: t.muted}} : undefined}, ax);
    }
    return lay;
  }
  function panelTitle(lay, i, text, t) {
    var s = i === 1 ? '' : String(i);
    lay.annotations.push({text: text, xref: 'x' + s + ' domain', yref: 'y' + s + ' domain', x: 0, y: 1.02,
      xanchor: 'left', yanchor: 'bottom', showarrow: false, font: {color: t.fg, size: 12}});
  }
  function drawNs(d) {
    var el = document.getElementById('cafs-ns'), t = theme(), traces = [];
    var lay = base(t, el, 2, 'ns per element (median of 5)', 'log');
    d.panels.forEach(function (p, i) {
      var s = i === 0 ? '' : String(i + 1);
      panelTitle(lay, i + 1, p.title, t);
      lay['yaxis' + s].tickvals = [2, 5, 10, 20, 50]; lay['yaxis' + s].ticktext = ['2', '5', '10', '20', '50'];
      Object.keys(NAME).forEach(function (a) {
        traces.push({x: p.K, y: p.series[a].median, xaxis: 'x' + s, yaxis: 'y' + s, name: NAME[a], legendgroup: a,
          showlegend: i === 0, mode: 'lines+markers', line: {color: COL[a], width: 2.5}, marker: {size: 6},
          customdata: p.series[a].min.map(function (v, j) { return [v, p.series[a].max[j]]; }),
          hovertemplate: 'K = %{x}<br>%{y:.2f} ns/element (median)<br>range %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra>' + NAME[a] + '</extra>'});
      });
    });
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function drawRatio(d) {
    var el = document.getElementById('cafs-ratio'), t = theme(), traces = [];
    var lay = base(t, el, 2, 'CAFS time ÷ hash-map-count time', 'log');
    d.panels.forEach(function (p, i) {
      var s = i === 0 ? '' : String(i + 1);
      panelTitle(lay, i + 1, p.title, t);
      lay['yaxis' + s].tickvals = [0.25, 0.5, 1, 2, 4]; lay['yaxis' + s].ticktext = ['0.25×', '0.5×', '1×', '2×', '4×'];
      lay['shapes'] = (lay['shapes'] || []).concat([{type: 'line', xref: 'x' + s + ' domain', x0: 0, x1: 1, yref: 'y' + s, y0: 1, y1: 1,
        line: {color: t.muted, width: 1, dash: 'dash'}}]);
      Object.keys(p.series).forEach(function (dist) {
        traces.push({x: p.series[dist].K, y: p.series[dist].ratio, xaxis: 'x' + s, yaxis: 'y' + s, name: DNAME[dist],
          legendgroup: dist, showlegend: i === 0, mode: 'lines+markers', line: {color: COL[dist], width: 2.5}, marker: {size: 6},
          hovertemplate: 'K = %{x}<br>CAFS takes %{y:.2f}× the hash-map count time<extra>' + DNAME[dist] + '</extra>'});
      });
    });
    Plotly.react(el, traces, lay, {responsive: true, displayModeBar: false});
  }
  function load(url) { return fetch(url).then(function (r) { return r.json(); }); }
  function go() {
    Promise.all([load('/lab/cafs-is-counting/ns-per-element.json'), load('/lab/cafs-is-counting/cafs-vs-count.json')]).then(function (v) {
      function all() { drawNs(v[0]); drawRatio(v[1]); }
      all();
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', all);
      var el = document.getElementById('cafs-ns'), narrow = el.clientWidth < 560;
      window.addEventListener('resize', function () { if ((el.clientWidth < 560) !== narrow) { narrow = !narrow; all(); } });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
</script>

**Uniform keys.**

- **`uint64`:** from K = 16 up, CAFS (blue) and a plain hash-map count (orange)
  track each other within about 30%. From K = 16 to 16k both beat pdqsort (green)
  by about 1.6–3.2×, close to the paper's 1.7–3.1×. CAFS's lead at the left edge is the ≤8-key
  path, which never touches a hash table.
- **`int32`:** CAFS sits lower. Against the better of my two fast counters it is
  1.1–1.7× faster in 8 of 12 mid-K cells (range 0.8–2.4× across uniform and
  Zipf). Part of the gap is `ankerl` being oddly slow
  on `int32` (more under limits).

| keys, distribution, K | pdqsort | hash-map count | CAFS, bucket swapped for hash map | CAFS |
|---|---:|---:|---:|---:|
| uint64, uniform, 8 | 4.75 | 3.72 | 2.74 | 2.64 |
| uint64, uniform, 1,000 | 12.6 | 4.24 | 4.91 | 4.67 |
| uint64, uniform, 16,384 | 16.9 | 5.74 | 6.05 | 5.60 |
| int32, uniform, 16,384 | 21.8 | 4.41 | 4.94 | 3.86 |
| uint64, Zipf, 65,536 | 15.7 | 7.35 | 8.40 | **14.0** |
| uint64, Zipf, 262,144 | 18.2 | 11.9 | 13.2 | **23.8** |
| int32, Zipf, 262,144 | 17.5 | 11.1 | 10.4 | **20.7** |

*ns per element, N = 10M. The last three columns come from one ablation run, so
they compare directly. pdqsort comes from the main grid. Its `int32` K = 16,384
cell is bimodal (two of five reps at ~15 ns), which is also the spike in the
`int32` panel above.*

<div id="cafs-ratio" style="width:100%;min-height:460px" role="img" aria-label="CAFS time divided by hash-map-count time versus distinct values K, for uniform, zipf and clustered-run keys, in two panels: uint64 and int32 keys. Below 1 means CAFS is faster."></div>

*CAFS time ÷ hash-map-count time, median of 5. Below 1× means CAFS is faster.
Hover for values.*

What the two charts and the ablation say:

- **For `uint64`, counting is the whole win.** From K = 16 to 16k, CAFS's time
  divided by the hash-map count's sits between 0.9 and 1.1 on uniform keys, and
  between 0.9 and 1.25 on Zipf.
- **The bucket mostly pays for CAFS's own overhead.** The bucket-swapped CAFS is
  up to 40% slower than real CAFS for `uint64` and 10–65% slower for `int32`, for
  K = 16 to 16k (on Zipf above that, the swap makes it faster; see below). It is
  also often slower than the plain count: the sampler, the run scan and the pair
  radix pass cost something on non-clustered data, and the bucket buys that back.
  For `int32` it buys back a little more.
- **The ≤8-key path is CAFS's clearest win.** At K ≤ 8 it is 1.2–1.3× faster
  than the better of my two counters for `uint64` and 1.1–1.5× for `int32`. At
  N = 1M, where writing the output costs less of the total, that grows to 2.0–3.8×
  and 1.0–2.0×. A hash table can't beat 8 compares with no hash and no branches.
- **Clustered runs favor CAFS, through run merging.** CAFS runs 1.2–1.7× faster
  than the better counter up to K ≈ 65k. The bucket-swapped version is just as fast,
  so the bucket isn't the reason. At "K = 1M" (really ~465k distinct values at this
  N), CAFS is bimodal. On four of five inputs the sampler sees only unique values
  and falls back to pdqsort, at 26–34 ns (`uint64`). On the fifth it takes the
  bucket path, at 12 ns.
- **Skewed data with many distinct values goes the other way.** On Zipf data
  with K ≥ 65k, CAFS is 1.5–2.2× slower than the hash-map count. At K = 262k it is
  slower than pdqsort, the sort it falls back to. At N = 1M the effect is smaller:
  1.1–1.4× slower for `uint64`, and none for `int32`.

The paper's "orders of magnitude" claim is about `std::unordered_map`, and it
didn't hold here either. Across all 176 grid cells (including a fourth input,
keys 0..K−1, where nothing in this post changes), CAFS ran between 0.8× (a
loss) and 8.4× the speed of a `std::unordered_map` count, with a median of 2.7×.
The largest ratios come from two places: K = 1M on 1M elements, where CAFS has
fallen back to pdqsort, and K ≤ 8, where it skips the table.

## Why Zipf breaks it

I re-ran CAFS's own estimator and sizing code on the inputs, then replayed its
bucket loop to count spills (`diag.cpp`):

| N = 10M, Zipf | true K | Chao1 estimate | table | elements spilled |
|---|---:|---:|---:|---:|
| K = 4,096 | 4,096 | 1,749 | 256 KiB | 4,794 (0.05%) |
| K = 65,536 | 65,536 | 3,200 | 512 KiB | 987,037 (9.9%) |
| K = 262,144 | 259,002 | 5,035 | 1 MiB | 1,592,338 (16%) |
| K = 1M | 763,005 | 6,037 | 1 MiB | 2,637,651 (26%) |

Chao1 extrapolates from how many values appear once or twice in the sample.
Under Zipf the sample is dominated by a few head values, so the long tail barely
shows up. With 1,024 samples the estimate stays under ~6k for every K I tested.

CAFS sizes its fixed table for that estimate. The tail overflows the buckets,
and up to a quarter of the array lands on the spill list, which is then sorted
with `std::sort`. The fallback never triggers: it needs more than half the array
spilled, and the worst case here is 26%.

A growable hash map doesn't care how wrong the estimate is. That's why swapping
the table fixes the Zipf rows in the ablation, while everything else in CAFS
stays the same.

On uniform data the estimate is fine, but 8 slots per value makes tables large.
At K = 262k with `uint64` keys it allocates 32 MiB. That fits in this machine's
260 MiB L3 but not in the 18 MiB L3 of the paper's i5. I didn't test how much
that matters there.

## What it means

If a column has fewer than ~10 distinct values, CAFS's unrolled path is a real
trick and worth stealing. It's 30 lines, and a hash map can't match it.

From there up to tens of thousands of distinct values, the speedup over general
sorts comes from counting. For 64-bit keys, a good hash map and a `fill_n`
already get it. Two things add to that: merging runs of equal neighbors before
the table, if your data arrives clustered, and, for 32-bit keys, CAFS's bucket.

Past that, check your skew. A table sized once from a sample is a bet that the
sample saw the distribution. Zipf-like data, which describes a lot of real
columns, loses that bet.

**Limits of this experiment:**

- One machine, a VM with no frequency pinning, and an L3 far larger than the
  paper's. Cells are 5 repetitions and can move 20–30% between runs. For example,
  `int32` uniform K = 1,000 reads 3.6 vs 2.6 ns (count vs CAFS) in the grid and
  3.1 vs 3.3 in the ablation run.
- No vqsort or IPS4o, which the paper also beats. I measured against counting, not
  against those.
- Synthetic inputs only, and one Zipf exponent.
- `ankerl::unordered_dense` is oddly slow on `int32` keys at small K (5.4 ns at
  K = 8, versus 3.6 ns for `uint64`). It reproduced across runs and I didn't chase
  it. My 30-line table does 2.0 ns there, so the `int32` counting baseline at tiny
  K is closer to CAFS than the chart suggests.

## Reproduce it

Everything is in
[`/code/cafs-is-counting/`](/code/cafs-is-counting/README.md), also as one
[tarball](/code/cafs-is-counting.tar.gz). That covers the benchmark (`bench.cpp`),
the ablation (`ablate.cpp`), the diagnostic (`diag.cpp`), the scripts, the raw
CSVs and the vendored headers (CAFS, pdqsort, ska_sort, unordered_dense), so
nothing needs downloading.

```sh
./run.sh               # full grid + ablation, ~45 min on 1 core
python3 analyze.py 64  # median tables (or 32)
python3 plot.py        # figures (needs matplotlib)
```

The Zipf result alone takes 10 seconds:

```sh
g++ -O3 -march=x86-64-v3 -std=c++20 -DNDEBUG -o ablate ablate.cpp
./ablate zipf 10000000 5 65536   # columns: dist,N,K,algo,rep,ns_per_elem,ok,key_bits
```
