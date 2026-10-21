---
title: "CAFS, minus the counting"
description: A new low-cardinality integer sort reports 1.7–3.1× over pdqsort. Most of that comes from counting, which a stock hash map also gives you. The AVX2 bucket adds 5–65% on top, and on skewed data with many distinct values CAFS runs about 2× slower than a plain hash-map count.
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
`count` times. The paper's additions sit around that core:

- **The bucket.** The table is an array of 64-byte buckets, one cache line each,
  holding 4 `uint64` keys (or 8 `int32` keys) and their counts. An AVX2 compare
  checks every key in a bucket at once. When a bucket is full the value goes to a
  spill list, which gets sorted with `std::sort` at the end.
- **The sampler.** Before allocating, CAFS reads 1,024 evenly spaced elements and
  estimates K with the Chao1 estimator. Chao1 counts how many sampled values
  appeared once or twice and extrapolates the unseen ones. The table gets about
  8 slots per estimated distinct value.
- **Two exits.** If the sample shows at most 8 distinct values, CAFS skips the
  table and counts with 8 unrolled, branch-free compares. If the estimate says K is
  more than N/2, it hands the array to a regular sort.

The paper reports 1.7–3.1× over pdqsort and 1.2–2.3× over Google's vqsort
when K ≪ N. That comes from 592,770 timed runs on a Core i5-12400F, with five
general-purpose sorts as the baselines.

## The question

None of those five baselines count. The paper's future-work section promises
a `std::unordered_map` hash-count baseline in v2 and says informal runs show CAFS
winning "by orders of magnitude". v2 doesn't include that baseline.

So the counting itself was never measured on its own. Three things I wanted to know:

1. How close does an ordinary hash-map count get to CAFS?
2. What does the AVX2 bucket add on top of counting? That is an ablation: same
   algorithm, different table.
3. What happens on skewed data? The paper's grid is uniform only, and its
   conclusion lists skewed inputs as an open question.

## Setup

Every counting contender shares the same 4-line emit step as CAFS, so only the
counting differs:

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

The contenders:

- **CAFS** from the authors' repo, with pdqsort as its fallback.
- **pdqsort**, **ska_sort** (a radix sort) and `std::sort`.
- **Three counting sorts:** `std::unordered_map`,
  [`ankerl::unordered_dense`](https://github.com/martinus/unordered_dense), and a
  30-line linear-probing table of my own.
- **The ablation, `cafs_hashmap`.** This is CAFS's own code with one change: the
  bucket table is replaced by `ankerl::unordered_dense`. The sampler, the ≤8-key
  path, run merging, the fallback rule and the final radix pass are all unchanged.

The inputs:

- **Key types:** `uint64` (the type the paper benchmarks) and `int32` (where a
  CAFS bucket holds 8 keys instead of 4).
- **Sizes:** N = 1M and 10M, with K from 2 to 1M.
- **Distributions:** uniform; Zipf with s = 1, where the most common value is
  about K times as frequent as the rarest; and clustered runs (uniform values in
  runs averaging 16 equal elements).

Every output is checked: the first repetition against `std::sort`, later ones for sortedness (the ablation checks every one against `std::sort`). The machine is a 4-vCPU KVM guest
on a 2.1 GHz Xeon with AVX-512, 2 MiB of L2 per core and a 260 MiB shared L3. I
used g++ 13.3 with `-O3 -march=x86-64-v3`. Each benchmark ran pinned to one core,
5 repetitions per cell, and I report medians. I couldn't pin the CPU frequency,
and the same cell can move 20–30% between separate runs. Treat gaps under about
1.2× as ties.

## Results

![ns per element vs distinct values, uniform keys, N=10M](/lab/cafs-is-counting/ns-per-element.png)

**Uniform keys.** In blue and orange, CAFS and a plain hash-map count are close
for `uint64` from K = 16 up to K = 1M. Against pdqsort (green) they win by the
same margin. CAFS's lead on the left edge is the ≤8-key path, which never touches
a hash table. For `int32` CAFS sits clearly lower, and that is where the bucket
earns something.

At N = 10M, in ns per element:

| keys, distribution, K | pdqsort | hash-map count | CAFS, bucket swapped for hash map | CAFS |
|---|---:|---:|---:|---:|
| uint64, uniform, 8 | 4.75 | 3.72 | 2.74 | 2.64 |
| uint64, uniform, 1,000 | 12.6 | 4.24 | 4.91 | 4.67 |
| uint64, uniform, 16,384 | 16.9 | 5.74 | 6.05 | 5.60 |
| int32, uniform, 1,000 | 10.9 | 3.07 | 4.01 | 3.25 |
| int32, uniform, 16,384 | 21.8 | 4.41 | 4.94 | 3.86 |
| uint64, Zipf, 65,536 | 15.7 | 7.35 | 8.40 | **14.0** |
| uint64, Zipf, 262,144 | 18.2 | 11.9 | 13.2 | **23.8** |
| int32, Zipf, 262,144 | 17.5 | 11.1 | 10.4 | **20.7** |

The last three columns come from one run, so they compare directly. pdqsort
comes from the main grid run.

![CAFS time divided by hash-map-count time, by distribution](/lab/cafs-is-counting/cafs-vs-count.png)

Reading the two charts:

- **Counting is most of the win.** For `uint64` with 16 ≤ K ≤ 16k, a hash-map
  count gets the same ~3× over pdqsort that CAFS does. CAFS and the counter are
  within noise of each other.
- **The bucket helps, modestly.** CAFS against its own hash-map variant shows the
  AVX2 bucket worth about 5–40% for `uint64` and 10–65% for `int32` at mid K.
  That is real, but the bucket isn't where the 3× comes from.
- **The ≤8-key path is CAFS's clearest win.** At K ≤ 8 it is 1.2–1.5× faster
  than a hash-map count for `uint64` and 1.8–3.2× faster for `int32`. A table
  can't beat 8 compares with no hash and no branches.
- **Clustered runs favor CAFS.** It merges equal neighbors before touching the
  table, so it runs 1.1–1.7× faster than a counter that doesn't, up to K ≈ 262k.
  At K = 1M it loses, though that cell's repetitions range from 10 to 27 ns.
- **Skewed data with many distinct values goes the other way.** On Zipf data
  with K ≥ 65k, CAFS is 1.5–2.2× slower than the hash-map count. At K = 262k it is
  slower than pdqsort, the sort it falls back to.

The paper's "orders of magnitude" claim is about `std::unordered_map`, and it
didn't hold here either. Across all 176 grid cells, CAFS beat a
`std::unordered_map` count by 0.8× to 8.4×, with a median of 2.7×. The largest
ratios come from two places: K = 1M on 1M elements, where CAFS has fallen back to
pdqsort, and K ≤ 8, where it skips the table.

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
shows up. With 1,024 samples the estimate tops out around 6k, whatever K is.
CAFS sizes its table for 6k values, the tail overflows the buckets, and up to a
quarter of the array lands on the spill list, which is then sorted with
`std::sort`. The fallback never triggers because spill only reaches 26%, not the
50% that would trip it.

A growable hash map doesn't care how wrong the estimate is. That's why swapping
the table fixes the Zipf column in the ablation, while everything else in CAFS
stays the same.

On uniform data the estimate is fine, but the 8-slots-per-value rule makes
tables large. At K = 262k with `uint64` it allocates 32 MiB. That fits in this
machine's 260 MiB L3 but not in the 18 MiB L3 of the paper's i5. I didn't test
how much that matters there.

## What it means

If your column has fewer than ~10 distinct values, CAFS's unrolled path is a
real trick and worth stealing. It's 30 lines, and a hash map can't match it.

From there up to tens of thousands of distinct values, the speedup over general
sorts is counting. The first thing to reach for is a good hash map and a
`fill_n`. CAFS adds a further 5–40% (`uint64`) or 10–65% (`int32`) with the AVX2
bucket, and more if equal values arrive in runs.

Past that, check your skew. A sampler-sized fixed table is a bet that the sample
saw the distribution. Zipf data, which describes a lot of real columns, loses
that bet.

**Limits of this experiment:**

- One machine, a VM with no frequency pinning, and an L3 far larger than the
  paper's. Cells are 5 repetitions and can move 20–30% between runs.
- No vqsort or IPS4o, which the paper also beats. I measured against counting, not
  against those.
- Synthetic inputs only, and one Zipf exponent.
- `ankerl::unordered_dense` is oddly slow on `int32` keys at small K (5.4 ns at
  K = 8, versus 3.6 ns for `uint64`). It reproduced across runs and I didn't chase
  it. My 30-line table does 2.0 ns there, so the "counting baseline" for `int32` at
  tiny K is closer to CAFS than the chart suggests.

## Reproduce it

The code, raw CSVs and the diagnostic are in
[`lab/cafs-is-counting/`](https://github.com/griffonn/maybenull.com/tree/main/lab/cafs-is-counting).
The headers are vendored, so no downloads are needed.

```sh
cd lab/cafs-is-counting
./run.sh              # full grid + ablation, ~45 min on 1 core
python3 analyze.py    # median tables
python3 plot.py       # figures (needs matplotlib)
```

The whole set took about 45 minutes of wall-clock here. The Zipf result alone
takes 10 seconds:

```sh
g++ -O3 -march=x86-64-v3 -std=c++20 -DNDEBUG -o ablate ablate.cpp
./ablate zipf 10000000 5 65536   # columns: dist,N,K,algo,rep,ns_per_elem,ok,key_bits
```
