---
title: "One-hash Bloom filters: where the false positives actually come from"
description: I simulated six Bloom filter layouts at equal memory, broke non-coprime partitions on purpose, and swapped gloom's divisions for multiplies. The one-hash trick is free. The cache-line block is not.
date: 2026-10-20
draft: false
---

Jim Calabro's [One Hash Bloom Filters](https://www.calabro.io/ohbf) (Oct 17) walks
through [gloom](https://github.com/jcalabro/gloom), the Go Bloom filter Bluesky
uses to keep mute, block and label checks away from Scylla. Gloom stacks three
ideas:

1. **One hash per key.** Gloom computes one 128-bit xxh3. The high half picks a
   block and the low half, folded to 32 bits, drives every probe.
2. **Cache-line blocks.** All `k` bits for a key fall inside one 512-bit,
   64-byte-aligned block, so a lookup costs at most one cache miss.
3. **Coprime partitions.** Each block is cut into `k` pieces with pairwise
   coprime sizes (for `k=7`: `61, 67, 71, 79, 83, 89, 62`). Probe `i` sets bit
   `h mod p_i` of piece `i`.

The post gives no false-positive numbers and no comparison against other
layouts. It also says the one-hash change made no measurable difference at
Bluesky, because lock contention was the real bottleneck. That leaves the
interesting questions open. What does each of the three ideas cost in false
positives? Is coprimality actually required? And are the `k` integer divisions
per probe worth worrying about?

I measured all three. Everything below is my own measurement unless it's
attributed to someone else.

## Setup

I wrote one C file with six layouts, all given **the same number of bits per
key**:

| layout | how it picks `k` bit positions |
|---|---|
| classic | `k` independent 64-bit hashes over the whole array |
| Kirsch–Mitzenmacher | `h1 + i·h2` over the whole array ([2006](https://doi.org/10.1007/11841036_42)) |
| OHBF, non-blocked | one hash, `k` coprime partitions spanning the whole array (the scheme Calabro credits to Lu et al.) |
| blocked bit-slices | one 512-bit block, `k` 9-bit slices of the hash, no partitions |
| **gloom** | one 512-bit block, gloom's prime table, 32-bit folded hash, `h mod p_i` |
| SBBF | Parquet's [split block Bloom filter](https://github.com/apache/parquet-format/blob/master/BloomFilter.md): a 256-bit block, one bit per 32-bit word via salt multiplies |

The gloom path follows its source line for line
(commit `a1f9124`, the version after Calabro's post):

```c
uint64_t base = fastrange64(h_hi, num_blocks) * 512;   // Lemire range reduction
uint32_t x    = (uint32_t)h_lo ^ (uint32_t)(h_lo >> 32);
for (int i = 0; i < k; i++)
    set_bit(base + offset[i] + x % prime[i]);
```

For each layout and each budget of 8, 10, 16 and 20 bits per key, I inserted
1M random keys, tried every `k`, and queried 20M (8–10 bpk) or 100M (16–20 bpk)
keys that had never been inserted. At the best `k` for each layout, the
smallest cell has about 6,700 false positives, so the
ratios below are stable to about ±1–2%.

I also coded the analytical model: a Poisson mixture over block load, the same
form gloom uses in its own `EstimateFalsePositiveRate`. For gloom, the
simulation matched the model within 0.6% in every cell. For example, at 16 bpk
I measured `9.378e-4` and the model predicts `9.384e-4`.

## The results

False-positive rate at each layout's best `k`, shown as a multiple of a
classic Bloom filter with the same memory:

| bits/key | classic | K–M | OHBF non-blocked | blocked bit-slices | **gloom** | SBBF |
|---:|---:|---:|---:|---:|---:|---:|
| 8  | 2.16% (1.00×) | 1.00× | 1.00× | 1.08× | **1.09×** | 1.54× |
| 10 | 0.82% (1.00×) | 1.00× | 1.00× | 1.18× | **1.23×** | 1.54× |
| 16 | 0.046% (1.00×) | 1.00× | 1.00× | 1.84× | **2.04×** | 2.86× |
| 20 | 0.0067% (1.00×) | 1.01× | 1.01× | 2.95× | **3.58×** | 6.32× |

Three things stand out.

**The one-hash trick costs nothing.** Non-blocked OHBF matches classic within
noise at every budget. The reason is the Chinese remainder theorem: if the
partition sizes are pairwise coprime, then `h mod p_1, …, h mod p_k` behave
like independent uniform draws for a uniform `h`. One hash really does give you
`k` hashes.

**Almost all of gloom's penalty comes from the block.** Look at 20 bits/key.
Forcing all `k` probes into one cache line costs 2.95×. Partitioning that line
into prime-sized pieces adds only another 1.21× on top (3.58 / 2.95). The cause
of the block penalty is the uneven load across blocks. At 20 bpk a block holds
about 25.6 keys on average, but the Poisson tail puts 35 or more in about 4.4% of
blocks. FP grows convexly with load, so those crowded blocks dominate the
average. Gloom's docs say this, and its sizing code compensates by adding
blocks.

The table above holds memory fixed. Turned around, here is how many bits per
key each layout needs to hit a target (from the model):

| target FP | classic | gloom | extra memory |
|---|---:|---:|---:|
| 1%    | 9.6  | 10.0 | +4%  |
| 0.1%  | 14.4 | 15.8 | +10% |
| 0.01% | 19.2 | 22.7 | +19% |

A real `gloom.New(20_000_000, 1e-4)` allocated 947,743 blocks, or 24.3 bits per
key. That's slightly above the model because its growth loop steps in 12.5%
increments. The trade is clear: you give up a few percent of memory at 1% FP,
and close to a fifth at 0.01%, in exchange for one cache miss per lookup instead
of up to `k`.

**SBBF is the honest baseline for "fast".** Parquet's layout costs 1.5× to 6×
more false positives at the same memory. In return, each probe is a multiply
and a shift into a fixed word, which maps directly onto 256-bit SIMD.
In my scalar C run, negative lookups on a 1.25 MB filter took about 10 ns for
SBBF and 30–45 ns for the other layouts. Gloom sits in a different spot: it gets
close to unblocked accuracy at low `k`.

## Breaking coprimality on purpose

The coprime requirement can look like number-theory decoration, so I broke it.
At `k = 8` and 16 bits/key, I kept everything else in gloom's layout and changed
only the partition sizes. Every set sums to 512:

| partition sizes | coprime? | measured FP | vs. independent-probe model |
|---|---|---:|---:|
| `64 ×8` | no, all equal | **39.4%** | ~430× |
| `56,58,60,62,64,66,70,76` | no, all even | 0.222% | 2.4× |
| `57,60,63,63,65,66,68,70` | no, mixed shared factors | 0.390% | 4.3× |
| `37,47,53,61,67,71,79,97` (gloom's `k=8`) | yes | 0.106% | 1.00× |
| `53,59,61,63,65,67,71,73` | yes | 0.092% | 1.00× |

With equal sizes, every probe lands at the same offset in its piece, so `k`
probes carry one probe's worth of information. The all-even row is subtler and
easy to miss in code review. All eight sizes are distinct and look reasonable,
but every `h mod p_i` has the same parity as `h`. The eight probes share one bit
of hash, and FP more than doubles. You only need two sizes that share a factor
to start leaking.

Both coprime rows match the independence model to three digits. The 32-bit fold
doesn't hurt either. Each block can hold at most 2³² distinct probe patterns, so
two keys in the same block collide on all `k` probes with probability about
`load / 2³²`. That's about 10⁻⁸, far below any FP rate anyone configures.

## A cheap accuracy win in the partition table

Coprimality forces the sizes apart, but not as far apart as gloom's table puts
them. Its `k = 8` entry spans 37 to 97 because it uses only primes. Small pieces
fill up fast, and FP is a product across pieces, so the overfull small pieces
dominate. Composite numbers can still be pairwise coprime: 63 = 3²·7 and
65 = 5·13 share no factor with each other or with the primes around them. A short
search (minimizing spread, subject to pairwise coprimality and a sum of 512)
found tighter sets:

| k | gloom's table | tighter coprime set | FP gain at 10 / 16 / 20 bpk |
|---:|---|---|---|
| 6 | `61 … 103` | `77,81,83,85,89,97` | 2.8 / 3.9 / 4.3% |
| 7 | `61 … 89` | `67,69,71,73,76,77,79` | 2.0 / 3.0 / 3.3% |
| 8 | `37 … 97` | `53,59,61,63,65,67,71,73` | **7.8 / 12.6 / 14.4%** |
| 9 | `41 … 73` | `47,49,53,55,57,59,61,64,67` | 2.9 / 5.2 / 6.1% |
| 10 | `31 … 73` | `37,41,43,47,49,53,57,59,61,65` | 2.9 / 6.4 / 7.8% |
| 11 | `29 … 67` | `31,37,41,43,46,47,49,51,53,55,59` | 2.5 / 6.2 / 7.7% |

The `k = 8` row is confirmed by simulation in the coprimality table above (0.092% vs
0.106%). Gloom's sizing picks the FP-minimizing `k`, which is often 7, 9 or 11
rather than 8, so the realized gain is smaller: about 2.4% at 10 bpk, 5% at 16
and 8% at 20. It's still free, since the change is a table edit with no code
change. Gloom's existing tests already check that each row is pairwise coprime
and sums to 512.

## The divisions: real, but not where the time goes

Go can't strength-reduce `x % p` when `p` is loaded from a slice. Compiling
gloom with `-gcflags=-S` shows one `DIVL` per probe in `testWithHash`. The
standard fix is [Lemire–Kaser–Kurz fastmod](https://arxiv.org/abs/1902.01961):
precompute `M = ⌊(2⁶⁴−1)/p⌋ + 1` once per partition. Each probe then costs two
multiplies, and the result is exact for every 32-bit `x`:

```go
func fastmod(x uint32, m uint64, p uint32) uint32 {
	hi, _ := bits.Mul64(m*uint64(x), uint64(p))
	return uint32(hi)
}
```

I patched a local copy of gloom (about 30 lines). Its full test suite still
passed, including the golden serialization tests, plus an added check of fastmod
against `%` for every prime in the table. Median `TestString` time over 6–10
runs on a 4-vCPU Xeon VM, Go 1.26:

| scenario | `k` | DIVL | fastmod | change |
|---|---:|---:|---:|---:|
| L1-resident filter, hits | 7 | 24.5 ns | 24.3 ns | ~0% |
| L1-resident filter, hits | 11 | 35.7 ns | 32.3 ns | −10% |
| 20M keys (27 MB), misses | 7 | 94 ns | 85 ns | noise-level |
| 20M keys (61 MB), misses | 11 | 92 ns | 96 ns | noise-level |

Once the filter is larger than cache, the block fetch swamps the arithmetic, and
the run-to-run spread (±15%) is wider than any effect. This matches Calabro's
Amdahl's-law point from another angle. Hashing once instead of `k` times, and
dividing instead of multiplying, are both small next to one DRAM round trip.
Fastmod is worth doing for small, hot filters with high `k`. It won't change a
large filter's speed.

## What I'd take away

- **"One hash" is not the tradeoff.** Done right, with coprime sizes, a single
  hash costs nothing in accuracy. That's also why it saves little time:
  xxh3 on a short key is already cheap.
- **The block is the tradeoff.** You get one cache miss per lookup and pay for
  it in FP. The penalty is about 1.2× at 10 bits/key and about 3–3.6× at
  20 bits/key, or +4% to +19% memory at equal FP. If you need very low FP
  rates and lookups are rare, a classic filter is more memory-efficient. If
  lookups are hot and memory is cheap, blocked filters win.
- **Coprime means pairwise coprime.** "Distinct" isn't enough. Two
  even sizes are enough to start losing accuracy, and nothing crashes when it
  happens. If you write your own partition table, test FP empirically, not just
  correctness.
- **If you use gloom,** the `k = 8` row is the one to change. Fastmod is a
  modest win only for L1/L2-sized filters.

## Reproducing

The C harness is about 60 lines. Each layout is a `case` in one function, the
hash is splitmix64, and the analytical model is about 35 lines of Python using
the Poisson mixture above. To check the gloom numbers yourself, run
`go build -gcflags=-S` on gloom to see the `DIVL`. For FP, compare
`EstimateFalsePositiveRate(numBlocks, k, n)` against a loop over
never-inserted keys. The two agreed within 1% in every configuration I tried.

**Sources:** Jim Calabro, [One Hash Bloom Filters](https://www.calabro.io/ohbf);
[jcalabro/gloom](https://github.com/jcalabro/gloom) (source read at `a1f9124`);
Kirsch & Mitzenmacher, [Less Hashing, Same Performance](https://doi.org/10.1007/11841036_42);
Apache Parquet [Bloom filter spec](https://github.com/apache/parquet-format/blob/master/BloomFilter.md);
Lemire, Kaser & Kurz, [Faster Remainder by Direct Computation](https://arxiv.org/abs/1902.01961);
a [Bloom filter survey](https://arxiv.org/abs/2106.12189) covering the one-hashing variant.
