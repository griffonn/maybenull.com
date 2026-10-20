---
title: "One hash, seven divisions: measuring the one-hash Bloom filter"
description: I benchmarked gloom's cache-line one-hash Bloom filter against split-block Bloom filters, the design Parquet uses. At the same memory the false-positive rates match. Once the filter outgrows the private caches, though, one-hash is up to 1.6× slower, and the division turns out not to be the reason.
date: 2026-10-20
draft: false
---

Jim Calabro's post [One Hash Bloom Filters](https://www.calabro.io/ohbf) has been on the
lobste.rs front page this week. It tells how Bluesky's startup bottleneck turned out to
be a mutex around a Bloom filter. It then introduces
[gloom](https://github.com/jcalabro/gloom), a Go library that combines two ideas.
Every key's bits live in one 64-byte cache line, and all k bit positions come from a
**single** hash, reduced modulo k pairwise-coprime partition sizes (the "one-hashing"
scheme the post credits to Lu et al.). The post is honest about the result: switching to
one hash brought no noticeable speedup, because contention had been the real problem. It
also skips the false-positive math and doesn't benchmark the filter itself.

So I had two questions. What does one-hashing actually buy? And how does it compare with
the obvious alternative, the **split block Bloom filter** (SBBF), which also uses one
hash and one cache line but needs no division at all? Parquet
[specifies it](https://github.com/apache/parquet-format/blob/master/BloomFilter.md), and
Impala and Kudu use the same design. Everything below is my own measurement on gloom
commit [`a1f9124`](https://github.com/jcalabro/gloom/tree/a1f91245d36661b646d222ce07bfc2ec7382d0c8),
and the code is [linked at the end](#reproduce-it).

## The contenders

All of them take one 128-bit xxh3 hash per key. The high 64 bits choose a block, using
Lemire's multiply-shift instead of a modulo. Here is what each design does inside the
block:

```text
OHBF (gloom), k=7, one 512-bit block split into pairwise-coprime partitions:
 |--- 61 ---|--- 67 ---|--- 71 ---|--- 79 ---|--- 83 ---|--- 89 ---|-- 62 --|
   h % 61     h % 67     h % 71     h % 79     h % 83     h % 89     h % 62
 (h = 32-bit fold of the low 64 hash bits; one bit per partition)

SBBF-512, k=8, one 512-bit block = 8 × 64-bit words:
 | word0 | word1 | word2 | word3 | word4 | word5 | word6 | word7 |
   bit = (x * salt[i]) >> 26        (x = low 32 hash bits; one bit per word)

SBBF-256 (Parquet spec), 256-bit block = 8 × 32-bit words, bit = (x * salt[i]) >> 27
blocked512-k:  k positions anywhere in the 512-bit block, 9 hash bits each
standard:      unblocked, k positions across the whole array (Kirsch–Mitzenmacher)
```

The SBBF probe for word `i` is one multiply and one shift, and the word it touches is
just `i`. The OHBF probe for partition `i` needs a modulo by a runtime value, an offset
lookup, and a split of the bit position into word and bit, because partitions straddle
word boundaries.

## False-positive rate: a tie

Every variant gets exactly the memory gloom allocates for 1M keys at the target rate.
Each cell is the observed rate on 3 × 10M absent keys, with insert and query keys
disjoint. At 1% the relative standard error is about 0.2%.

| target (gloom's k, bits/key) | gloom | OHBF, 64-bit h | SBBF-512 k8 | SBBF-256 (Parquet) | blocked512-k | standard |
|---|---|---|---|---|---|---|
| 5% (k=5, 7.0) | 3.669% | 3.675% | 5.093% | 5.605% | 3.641% | 3.442% |
| 1% (k=7, 10.8) | 0.718% | 0.717% | **0.725%** | 0.893% | 0.686% | 0.566% |
| 0.1% (k=9, 16.2) | 0.0880% | 0.0889% | **0.0848%** | 0.124% | 0.0788% | 0.0426% |

What I take from this:

- **gloom's estimator is excellent.** Its sizing code models the blocking penalty as a
  Poisson mix of per-block loads, using the real partition sizes. It predicts 0.7155% at
  the 1% setting, and I measured 0.7178%. It deliberately oversizes, so a "1%" filter
  actually delivers about 0.72%.
- **At the same memory, SBBF-512 matches OHBF.** It is 1% worse at the 1% target and 4%
  better at 0.1%. The exception is the 5% target, where SBBF's fixed k=8 is too many
  probes for 7 bits per key. A variant that probes fewer words would close that gap.
- **The CRT shortcut costs nothing measurable.** The one-hash argument is that a single
  value reduced mod pairwise-coprime sizes gives independent positions, thanks to the
  Chinese Remainder Theorem. That holds only if the value spans the product of the
  sizes. For k=7 the product is 61·67·71·79·83·89·62 ≈ 1.05×10¹³, about 43.3 bits, but
  gloom divides a **32-bit** fold of the hash, so only 1 in 2,444 residue tuples can
  ever occur. I re-ran the scheme with the full 64-bit value. The difference is within
  noise in both directions (0.718% vs 0.717%, 0.0880% vs 0.0889%). A false positive only
  needs each partition's bit to land on a set bit, and the reachable tuples are still
  uniform on every partition and on every small group of them. My reimplementation
  matches the library bit for bit: the fp counts are identical.
- At the 1% and 0.1% targets, Parquet's 256-bit SBBF pays 23–46% more false positives
  than either 512-bit design at equal memory. Smaller blocks mean a noisier per-block load, which is the same
  blocking penalty gloom's estimator models. If you control the format, prefer
  cache-line-sized blocks.

So on accuracy, one-hashing neither wins nor loses. The question is speed.

## Speed: fine in cache, 1.6× slower out of it

These are single-threaded timings, pinned to one core of a 4-vCPU Xeon VM (2.1 GHz),
with Go 1.26. To keep key memory traffic out of the measurement, keys are hashed by an
inlined splitmix64. I report the median of 5 runs (3 for the largest size), in ns per
operation. "miss" means querying absent keys, "hit" means querying present keys, and the
filter is sized for 1% FPR.

| filter size | operation | OHBF (gloom) | OHBF + fastmod | SBBF-512 | SBBF-256 | blocked512-k | standard |
|---|---|---|---|---|---|---|---|
| 13 KB (n=10k) | add | 19.0 | 19.1 | 12.9 | 12.2 | 12.0 | 13.9 |
| | miss | 23.9 | 21.4 | 18.8 | 18.0 | 16.1 | 20.7 |
| | hit | 26.1 | 17.3 | 12.7 | 12.2 | 12.9 | 15.8 |
| 1.3 MB (n=1M) | add | 23.8 | 19.9 | 15.6 | 13.0 | 14.9 | 23.4 |
| | miss | 31.6 | 25.5 | 23.4 | 21.8 | 21.0 | 27.6 |
| | hit | 30.5 | 22.1 | 16.2 | 15.9 | 15.4 | 24.2 |
| 135 MB (n=100M) | add | 144 | 139 | 91 | 84 | 115 | 158 |
| | miss | 88 | 87 | 66 | 66 | 61 | 101 |
| | hit | 135 | 143 | 87 | 74 | 84 | 162 |

The real library, with its own xxh3 call on 8-byte keys, shows the same gap: at n=100M,
`gloom.Filter` took 152–162 ns per add, 103–117 ns per miss and 149–184 ns per hit. SBBF-512
fed by the same xxh3 took 89–90, 73–99 and 89–113 ns
([`lib.txt`](/code/one-hash-bloom-vs-split-block/lib.txt)).

In cache, OHBF runs 1.3–2× slower than SBBF, and most of that gap is division. Swapping
the `%` for Lemire's
[fastmod](https://arxiv.org/abs/1902.01961), with one precomputed 64-bit constant per
partition, produces bit-identical positions (the repro includes a test for this). It cut
hit latency from 26 to 17 ns on the 13 KB filter.

At 135 MB the picture changes. OHBF is about 1.6× slower on adds and 1.55× on hits,
and **fastmod does nothing**. Division is no longer the bottleneck.

## Why: it's the work per lookup, not the divide

One expensive cache miss per lookup should make every blocked design run at the same
speed. They don't, because a CPU hides memory latency by keeping several independent
lookups in flight. How many it can track depends on how many instructions each lookup
puts into the out-of-order window. The disassembly (`go tool objdump`) of the fastmod
OHBF `Has` shows about 27 instructions per probe. Each probe has two bounds checks, loads
the offset and the magic constant from slices, does two multiplies, and splits the
position into word and bit. With k=7 that's roughly 190 instructions per hit. Going by
the source, an SBBF probe is a multiply, a shift and a test, on a word whose index is
known before any arithmetic happens. The standard filter is slowest of all, because it
takes up to 7 separate misses.

I had no access to perf counters on this VM, so I tested that explanation indirectly. If
memory-level parallelism is the limit, then issuing all the misses up front should erase
most of the gap. So I added a batched lookup. It computes the block index for 16 keys,
touches each block with a plain load, and only then runs the normal probes:

| 135 MB filter, batched ×16 | miss | hit |
|---|---|---|
| OHBF (gloom) | 51–54 | 49–61 |
| OHBF + fastmod | 48–52 | 46–53 |
| SBBF-512 | 42–45 | 47–49 |

Batching takes OHBF hits from 135 to about 50–60 ns, a 2.3–2.7× improvement, while SBBF
gains about 1.8×. The 1.6× gap shrinks to 1.0–1.25×. That fits the explanation: unbatched,
OHBF's longer per-lookup instruction stream limits how many misses overlap. This is an
inference from timings, not a hardware-counter measurement. A caveat on "out of cache":
this VM reports a 260 MB shared L3, so the 135 MB filter is past the private 2 MB L2
but probably partly L3-resident. On a box where it truly lives in DRAM, I'd expect the
gap to widen, not close.

## So what should you use?

- **One-hashing's selling point, avoiding k hash computations, doesn't apply here.**
  xxh3-128 on an 8-byte key took 3 ns in my tests. Any modern design hashes once and
  derives the rest, and SBBF derives its probes more cheaply than modular partitions
  do.
- **At equal memory and the usual 0.1–1% targets, SBBF-512 gives the same FPR, inserts
  faster, and stays up to 1.6× faster once the filter outgrows the private caches.**
  Partitions do help in one place: they let gloom pick any k from 3 to 17, which
  matters at loose targets like 5%.
- **If you keep OHBF**, two cheap wins came out of this. Use fastmod, or constant
  divisors per k, which speeds up in-cache hits by up to 1.5×. And offer a batch lookup API, which
  helps 2.5× at scale.
- **None of this changes Calabro's conclusion.** In Bluesky's deployment the cost was
  lock contention, and that's where the 6× deploy-time win came from. These are
  single-thread differences of tens of nanoseconds, which matter only once the filter is
  the hot loop.

## Caveats

- Go only, single-threaded. A C++ or Rust SBBF would use AVX2 to do all 8 probes in a handful
  of vector instructions, which would likely widen the gap. Go's compiler did not vectorize anything here.
- The cross-design speed numbers come from my reimplementation of the gloom scheme. Its
  FPR is bit-identical to the library, and the library-level benchmark shows the same
  or a slightly larger gap (about 1.6–1.8× on adds and hits at n=100M).
- One shared cloud VM. I report medians, and repeat runs moved individual cells by about
  ±15%. The ordering never changed.
- 8-byte keys only. Longer keys add the same hashing cost to every design.

## Reproduce it

The code is in [`/code/one-hash-bloom-vs-split-block/`](https://github.com/griffonn/maybenull.com/tree/main/public/code/one-hash-bloom-vs-split-block).
It's a single Go module that pins gloom `a1f9124`. `go test` checks fastmod against `%`
across every k gloom supports. `go run . fpr -fp 0.01` gives the FPR table, and `speed`,
`batch` and `lib` give the timing tables. The exact output behind this post is in
`fpr.txt`, `speed.txt`, `batch.txt` and `lib.txt`.

*Sources: [Jim Calabro, "One Hash Bloom Filters"](https://www.calabro.io/ohbf),
[jcalabro/gloom](https://github.com/jcalabro/gloom/tree/a1f91245d36661b646d222ce07bfc2ec7382d0c8),
[Parquet Bloom filter spec](https://github.com/apache/parquet-format/blob/master/BloomFilter.md),
[Lemire et al., "Faster Remainder by Direct Computation"](https://arxiv.org/abs/1902.01961).
Benchmarks, analysis and conclusions are my own.*
