---
title: "One-hash Bloom filters, taken apart"
description: An ablation of the prime-partition "one hash" trick. It costs accuracy and speed, and a 128-bit hash already gives you the same thing for free.
date: 2026-10-20
draft: false
---

[One Hash Bloom Filters](https://calabro.io/ohbf) was on the Hacker News front page
today. It describes [gloom](https://github.com/jcalabro/gloom), the Go Bloom filter
behind some of Bluesky's hot paths. Each key is hashed once, and the *k* probe
positions come from reducing that one hash modulo *k* pairwise-coprime partition
sizes inside a 64-byte block. The technique comes from a paper by Lu et al. The
post is upfront that it has no numbers for the one-hash part on its own, and that
in Bluesky's workload the real win came from fixing contention.

So I ran that ablation. I kept everything about gloom's design fixed and changed only
the way the *k* positions are derived. Then I measured false-positive rate and lookup
cost. Short version: the trick works, and gloom's error model is very accurate. But
you pay for it in both accuracy and speed, and the thing it saves (extra hashing) was
never being spent in the first place.

## What gloom actually does

From the source (`hash.go`, `params.go`, `bloom.go`):

```
key ──xxh3-128──► Hi (64 bits) ──Lemire reduce──► block index (512-bit block)
                  Lo (64 bits) ──fold to 32 bits──► x
                                                    │
     k=7 partitions: 61 | 67 | 71 | 79 | 83 | 89 | 62   (sum = 512)
     bit_i = offset_i + x mod p_i
```

The partition sizes are pairwise coprime. By the Chinese Remainder Theorem, that means
the residues `x mod p_i` act as independent coordinates rather than as copies of one
another.

Notice that gloom already hashes with a **128-bit** xxh3 and then throws away half of
`Lo` by folding it to 32 bits. A 512-bit block needs 9 bits per probe, and 7 × 9 = 63.
So `Lo` alone holds seven independent probe positions with no arithmetic at all. That
gives us the obvious control: the same blocks and the same block selection, with
positions taken as plain 9-bit slices of the hash.

## The setup

Four variants share the same hashes and the same memory budget (`n × bits_per_item`).
The three blocked ones also share the same 512-bit blocks:

| variant | probe positions | divisions per probe |
|---|---|---|
| **prime-onehash** | gloom's scheme, reimplemented on the same hash values | 1 (`DIV r32`) |
| **blocked-slice** | k unpartitioned 9-bit slices of `Lo` | 0 |
| **sbbf512** | split-block (Parquet/Impala style): one bit in each of the 8 words, `(x·salt_i) >> 26` | 0 |
| **classic** | unblocked, Kirsch–Mitzenmacher `h1 + i·h2` over the whole array | 0 |

```go
// prime-onehash (what gloom does)
x := uint32(h.Lo) ^ uint32(h.Lo>>32)
pos := f.offsets[i] + x%f.primes[i]

// blocked-slice (the control)
pos := v & 511; v >>= 9          // v starts as h.Lo
```

I inserted n = 1M sequential keys and queried 20M keys that were never inserted. The
machine is a 4-vCPU Xeon @ 2.1 GHz with Go 1.24. To make sure my reimplementation is
faithful, I also ran the actual gloom library: at k=7 and k=10 it produced
**exactly** the same false-positive counts as my version.

## Result 1: gloom's error model is very good

gloom does not use the textbook `(1 − e^(−kn/m))^k`. Instead it estimates the FPR as a
Poisson mixture over block loads, multiplied out across the real prime partition sizes.
That model holds up:

| bits/item | k | measured (gloom lib) | gloom's estimate |
|---|---|---|---|
| 10 | 7 | 1.00889% | 1.00839% |
| 10 | 10 | 1.40844% | 1.41299% |
| 20 | 7 | 0.03329% | 0.03368% |
| 20 | 10 | 0.02456% | 0.02450% |

Agreement is within 0.5% relative at 1% FPR. A lot of Bloom filter libraries quietly
miss their configured rate, so this is worth noting. Its `OptimalParams` also grows
the block count to make up for the blocking penalty, which most libraries don't do.

## Result 2: the one-hash trick costs false positives

Here is the best k for each variant at equal memory, measured over k = 5…14:

| bits/item | prime-onehash | blocked-slice | sbbf512 (k=8) | classic |
|---|---|---|---|---|
| 10 | 1.009% (k=7) | 0.966% (k=6) | 1.049% | 0.819% (k=7) |
| 20 | 0.0238% (k=11) | 0.0197% (k=11) | 0.0264% | 0.0064% (k=13) |

The penalty is about 4% at 10 bits/item and about 20% at 20 bits/item. The numerical
model shows where it comes from. I evaluated the same Poisson-mixture formula for three
layouts of one block:

| bits/item, k | prime partitions | equal partitions (512/k) | unpartitioned | classic |
|---|---|---|---|---|
| 10, k=7 | 1.0086% | 0.9838% | 0.9571% | 0.8194% |
| 20, k=10 | 0.0245% | 0.0210% | 0.0197% | 0.0089% |

There are two separate costs:

1. **Partitioning.** Each key sets exactly one bit per partition, which constrains the
   layout more than k free positions do. This is the known gap between partitioned and
   standard Bloom filters.
2. **Unequal partitions.** To keep sizes coprime and summing to 512, gloom's k=10 layout
   runs from 31 to 73 bits. The small partitions fill up faster, and the FPR is a
   *product* across partitions, so the fullest ones dominate. This cost grows with k,
   and low target rates call for large k. That's why the gap widens at 20 bits/item.

I also checked a third possible problem, and it turned out not to matter. For k=7 the
CRT product of the partition sizes is about 1.05·10¹³ ≈ 2^43.3, but `x` holds only 32
bits. So only one in about 2,400 of the possible position tuples can ever be produced.
That sounds alarming, but the measured FPR matches the model, which assumes fully
independent partitions, to three significant figures. A 32-bit dividend is enough in
practice. gloom's code comment reaches the same conclusion with a per-partition
argument; I'd base it on the joint measurement instead.

## Result 3: the divisions aren't free

Here is ns per `Test` (best of 5 runs over 20M lookups). "neg" means keys that were
never inserted and exit early; "pos" means inserted keys that check all k bits:

| filter size | prime-onehash k=7 | blocked-slice k=7 | sbbf512 |
|---|---|---|---|
| 3K keys (in L1/L2), neg | 21–24 | 13–14 | 14–17 |
| 3K keys, pos | 16 | 13 | 9–11 |
| 30M keys (37.5 MB), neg | 41–44 | 27–28 | 31–33 |
| 30M keys, pos | 47–57 | 38–40 | 33–35 |

The prime scheme is 1.2–1.7× slower per lookup. Every probe is a hardware `DIV`,
because the divisor is loaded from a slice and isn't a compile-time constant. What
surprised me is that the gap stays just as large when the filter is far bigger than
cache. You'd expect a cache miss (~100 ns) to hide a few divides. My guess, which I
haven't verified with perf counters, is that the divides fill the out-of-order window
and leave less room to overlap misses from neighbouring lookups. A batch lookup
benchmark would be the way to test that.

For gloom's production use this hardly matters. The post itself says lock contention
was the bottleneck. But "one hash" is pitched as a speed technique, and on this
hardware it costs time rather than saving it.

## Result 4: the thing that actually costs you is blocking

Look at the classic column again. At 20 bits/item, the unblocked filter is **3×
more accurate** than any of the blocked variants with the same memory. Putting every
probe in one cache line is what buys the guaranteed single miss, and it's a deliberate
tradeoff. But how you pick positions *inside* the block is a rounding error next to
it:

```
FPR at 20 bits/item (lower is better)
classic          ██████▍                 0.0064%
blocked-slice    ███████████████████▋    0.0197%
prime-onehash    ███████████████████████▊ 0.0238%
sbbf512          ██████████████████████████▍ 0.0264%
```

## So when is one-hashing the right call?

The Lu et al. idea fits a setting where hashing is expensive and you get one hash
value of limited width, and the alternative is computing k independent hashes. It
doesn't fit a setting where a single xxh3-128 call already gives you 128 good bits.
Those 128 bits cover up to 14 nine-bit probes into a 512-bit block. In the
experiments above, the best k never went past 13.

If you're building a blocked Bloom filter today, here's what I'd do based on this:

- Hash once with a wide hash (64 or 128 bits). Take the block from the high bits with
  a multiply-shift, and take the in-block positions as **bit slices** of the rest.
  It's simpler, has no divides, and gives a lower FPR than coprime partitions.
- If you want SIMD-friendly code with one bit per word, use split-block (SBBF). It's
  slightly less accurate per bit but very fast on hits.
- If your target is below about 0.1% and memory matters more than one cache miss, a
  blocked filter is the wrong structure. Use a classic filter, or better, a
  [binary fuse or xor filter](https://arxiv.org/abs/2201.01174) for static sets.
- Do copy gloom's habit of modeling the blocked FPR honestly and sizing to it.

## Caveats

- This measures a single core in Go on one VM. Division latency varies a lot by
  microarchitecture: it's much cheaper on recent Intel and AMD cores than on older
  ones, so the speed gap may be smaller on your hardware.
- My prime-onehash numbers come from a reimplementation, but I validated it against
  the gloom library (identical FP counts), and the library's design comes straight
  from its source.
- Sequential integer keys hashed with xxh3. With a weaker hash, the comparison could
  come out differently.

*Sources: [calabro.io/ohbf](https://calabro.io/ohbf),
[jcalabro/gloom](https://github.com/jcalabro/gloom) (read at HEAD on 2026-10-20).
All measurements, models and conclusions above are my own.*
