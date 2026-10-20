---
title: "tsrs vs tsgo: where the 8× actually comes from"
description: I raced Cloudflare and VoidZero's Rust port of the TypeScript compiler against the Go one on a 4-core box, then turned off tsrs's extras one at a time. Rust gets you about 2.5×. The other 3× is scheduling and laziness, and none of that needs Rust.
date: 2026-10-20
draft: false
---

Today Cloudflare [announced tsrs](https://blog.cloudflare.com/typescript-rust-port/), a Rust port
of the TypeScript 7 compiler, built with VoidZero and incubated in the
[Oxc](https://github.com/oxc-project/tsrs) toolchain. TypeScript 7 is itself the Go port
("tsgo"), which shipped in July with roughly a 10× speedup over the old JavaScript `tsc`.
Cloudflare's headline number is up to 8× faster than TypeScript 7.0 at equal worker counts,
with about 70% less memory.

The obvious reading is "Rust beats Go by 8×". That isn't really what's going on. tsrs ships
two algorithmic changes that are switched on by default, and it has flags to turn both off.
That made it possible to split the speedup into a part that comes from the port itself and a
part that comes from doing the work differently. Everything below is my own measurement unless
I say otherwise.

## Setup

- **Machine:** 4 vCPU Intel Xeon @ 2.1 GHz, 15 GB RAM. That's closer to a laptop or a
  standard CI runner than the 16 to 64 vCPU boxes in tsrs's
  [own results](https://github.com/oxc-project/tsrs/blob/main/bench/RESULTS.md).
- **Compilers:** `typescript@7.0.2` (tsgo, current stable), `typescript@7.1.0-beta` (tsgo
  beta) and `tsrs@0.18.0`, which ports a TypeScript 7.1-dev commit from 30 September. I
  called the native binaries directly so no Node launcher is in the timing.
- **Projects** (none are in tsrs's published benchmark set):
  - **type-fest**: 518 files of type-level code plus its `test-d` suite. Very heavy on
    instantiation. It's the stress test.
  - **zod** (`packages/zod`): 332 files, a mix of library code and tests.
  - **TypeScript 5.9's `src/compiler`**: 192k lines of plain, mostly non-generic code.
- **Method:** `--noEmit`, median of 3 runs (5 for the compiler project, 1 for a few slow
  tsgo 7.0 configs). I measured wall time, CPU time (user plus sys) and peak RSS.

## Correctness first

Speed doesn't matter if the answers differ, so I checked that before timing anything.

- **TS 5.9 compiler:** 27 diagnostics, including real `TS2322`/`TS2345` type errors. The
  sorted output of tsgo 7.0, tsgo 7.1 and tsrs (at 1 checker and at 4) was byte-identical.
- **zod:** 28 diagnostics, byte-identical between tsgo 7.1 and tsrs.
- **type-fest:** zero errors from every compiler in every mode. That's a stronger check
  than it sounds. The test suite has 567 `@ts-expect-error` directives, and each one turns
  into a `TS2578` error if the compiler fails to report the error it expects. So "0 errors"
  means tsrs found exactly the errors tsgo found, at 567 specific locations, including with
  `--noLazyMembers` off and on.

That's a small sample. tsrs's own docs report 13,458 of 13,462 conformance baselines
matching, and I didn't find a counterexample.

## The headline, reproduced

Default settings on all three. On a 4-core machine both compilers end up with 4 checkers,
so this is already an equal-worker comparison.

| project | tsgo 7.0 | tsgo 7.1β | tsrs | tsrs vs 7.0 | tsrs vs 7.1β |
|---|---|---|---|---|---|
| type-fest | 34.8 s / 8.9 GB | 22.1 s / 6.4 GB | **2.75 s / 1.5 GB** | 12.6× | 8.0× |
| zod | 5.11 s / 1.4 GB | 3.68 s / 1.1 GB | **0.71 s / 0.3 GB** | 7.2× | 5.2× |
| TS compiler | 2.23 s / 482 MB | 1.68 s / 298 MB | **0.47 s / 143 MB** | 4.7× | 3.6× |

The claim holds up on a small machine. If anything, type-fest beats it. Two things also
stand out:

1. **7.1 beta is already 1.3 to 1.6× faster than 7.0.** Cloudflare's post says some of
   7.1's optimizations came from tsrs.
2. **The gap grows with how type-heavy the code is.** It's 3.6× on plain compiler code and
   8× on type-fest.

## Step 1: one thread, same algorithm

Start by removing parallelism. With `--singleThreaded` there's no scheduling to differ on.

| project | tsgo 7.1β | tsrs | ratio |
|---|---|---|---|
| type-fest | 27.6 s | 11.1 s | 2.5× |
| zod | 5.71 s | 1.95 s | 2.9× |
| TS compiler | 2.86 s | 1.32 s | 2.2× |

Both compilers report their work with `--extendedDiagnostics`. On type-fest single-threaded:

```
                 tsgo 7.1β       tsrs        tsrs --noLazyMembers
Instantiations:  18,062,470    18,026,279    18,059,075
Symbols:         16,202,969     7,565,955    16,201,629
Types:            5,939,581     1,326,897     1,331,570
Memory used:      3,565,736K    1,160,268K    1,828,196K
Check time:          27.2 s        11.4 s        12.3 s
```

The instantiation counts agree to within 0.2%, so this is the same type-level work done at
a different cost per step. That works out to about **1.5 µs per instantiation in Go and 0.63
µs in Rust**. Call it 2.2 to 2.9× per core, which is what I'd guess for a careful port with
arenas and compressed handles (the tsrs
[engineering notes](https://github.com/oxc-project/tsrs/blob/main/notes/README.md) cover
pointer compression and per-file arena regions). Two caveats: the 7.1β binary is a few
commits away from the one tsrs ports, and I can't tell whether the two compilers count
"Types" the same way, so don't read much into that 4.5× row.

**Is it Go's GC?** Mostly not, at least in wall time. I reran tsgo single-threaded with
different `GOGC` settings:

| type-fest, 1 thread | wall | CPU | RSS |
|---|---|---|---|
| `GOGC=100` (default) | 27.6 s | 35.2 s | 4.6 GB |
| `GOGC=400` | 26.9 s | 30.9 s | 4.6 GB |
| `GOGC=off` (11 GiB soft limit) | 26.7 s | 26.6 s | 5.9 GB |

Turning the collector off saves about 25% of CPU time but only about 3% of wall time. The
collector runs on cores the single checker isn't using, so it mostly burns spare capacity
rather than delaying the result. At 4 checkers, where every core is busy, `GOGC=off` helped
more: 22.4 s down to about 19.5 s. That's still roughly 7× behind tsrs, so GC is real but
it's a minor term.

## Step 2: four threads, and where Go stops scaling

Here is wall time on type-fest as I add checkers:

```
checkers      1      2      4      8
tsgo 7.1β   28.4   26.8   22.4   25.9   s
tsrs        10.0    6.0    2.8    4.0   s
```

tsrs scales 3.6× from 1 to 4 checkers. tsgo manages 1.27×. Its total CPU time also nearly
doubles over the same range, from 37.8 s to 67.8 s, while tsrs's barely moves (9.8 s to 10.7
s).

You might guess that tsgo's checkers keep redoing each other's work. Each checker has its own
type caches, so shared generic types get instantiated more than once. That does happen, but
it happens to both compilers about equally. At 4 checkers, instantiations rise from about
18.0M to 20.65M in tsgo and 20.35M in tsrs, about 13% duplicated work either way. Duplication
doesn't explain the difference.

## Step 3: turn off tsrs's extras

tsrs has two flags that reproduce tsgo's behaviour:

- `--checkerAssignment go` uses tsgo's static assignment of files to checkers. tsrs's
  default instead groups files by directory subtree and lets idle checkers steal work
  ([DEFAULTS.md](https://github.com/oxc-project/tsrs/blob/main/docs/DEFAULTS.md)).
- `--noLazyMembers` turns off lazy member instantiation. tsgo eagerly builds an
  instantiated type's whole member table. tsrs builds members only when something asks for
  them.

On type-fest at 4 checkers:

| tsrs mode | wall | CPU | RSS |
|---|---|---|---|
| default | 2.9 s | 10.4 s | 1.53 GB |
| `--noLazyMembers` | 4.0 s | 14.5 s | 2.36 GB |
| `--checkerAssignment go` | 5.4 s | 10.2 s | 1.45 GB |
| both (tsgo-equivalent) | 8.4 s | 15.0 s | 2.21 GB |
| *tsgo 7.1β, for reference* | *22.4 s* | *67.8 s* | *6.2 GB* |

Each flag does something different.

**Checker assignment costs wall time but not CPU.** With Go's assignment, tsrs does the same
10 s of work but takes 5.4 s instead of 2.9 s, because cores sit idle. On type-fest a static
split leaves some checkers holding much more expensive files than others. Work stealing fixes
that. The effective parallelism falls from about 3.6 cores to about 1.9.

**Lazy members save real work and memory.** tsrs's counters show why. Across the type-fest
run, instantiated object types declared 13.6 million members, and only 3.3 million (24%) were
ever instantiated. The checker mostly asks narrow questions: does this property exist, what's
its type, is there an index signature. Lazy tables answer those without building the whole
mapped type. In this run 758,500 conditional mapper compositions were avoided outright.
Single-threaded, lazy members cut the symbol count from 16.2M to 7.6M and the checker's memory from 1.83 GB to 1.16 GB. tsrs's docs warn this can change diagnostics near the
instantiation depth limit. It didn't in any of my projects.

With both flags set, **tsrs is 2.7× faster than tsgo on type-fest**, close to the
single-threaded ratio. The rest of the 8× comes from those two changes:

```
type-fest, 4 cores, tsgo 7.1β → tsrs default (8.0×)

  same algorithm, Rust vs Go           ≈ 2.7×   (22.4 s → 8.4 s)
  × lazy members + work stealing       ≈ 3.0×   ( 8.4 s → 2.8 s)
```

On zod the split is 4.4× for the port and 1.4× for the algorithmic changes: zod is less
type-heavy, so laziness matters less, and scheduling has less imbalance to fix. On plain
compiler code it's almost all port.

## What this means

**Most of the speedup on type-heavy code isn't Rust.** Work stealing and lazy member tables
are ordinary algorithmic changes that would fit in Go. tsgo's checkers would still not share
caches, but tsrs's don't either. 7.1 has already taken some tsrs optimizations, and these
two look like obvious next ones. If tsgo picks them up, the gap on type-fest should shrink
from about 8× to about 2.5 to 3×, which matches what I measured per core.

**What's left for Rust is per-core speed and memory.** That's a consistent 2.2 to 2.9× per
core and 2 to 4× less memory in every mode. Even in tsgo-equivalent mode, tsrs peaked at 2.2
GB on type-fest where tsgo 7.1 peaked at 6.2 GB. Turning Go's GC off doesn't close it: the
heap just grows to 7.8 GB. On a 16 GB laptop running an editor, a browser and a language
server, that difference decides whether the machine swaps.

**Small machines see the full effect.** tsrs's published numbers are from 16-vCPU and larger
boxes, and its default gets up to 16 checkers there. On 4 cores both compilers use 4, and the
speedup on type-heavy code was still 8×. That's because tsgo hits a scaling wall early: going
from 1 to 4 checkers buys it only 1.27×, because its static assignment and GC overhead eat
most of the benefit of extra cores. Its 8-checker run was slower than its 4-checker run. (Running 8 checkers on 4 cores slows tsrs down too.)

**Caveats.** This is one machine, three projects and a beta compiler. tsrs is labelled
experimental and has gaps (`--init`, `--showConfig`, `--locale` and `--generateTrace` aren't
supported yet). Its default file assignment also makes diagnostic order deterministic, which
tsgo's isn't, so a plain unsorted diff of the two outputs can differ even when both are
correct. I compared sorted output.

## Reproduce it

```sh
npm i ts70@npm:typescript@7.0.2 ts71@npm:typescript@7.1.0-beta tsrs@0.18.0-ts7.1.0-dev.20260929

# native binaries
GO71=node_modules/ts71/node_modules/@typescript/typescript-linux-x64/lib/tsc
RS=node_modules/@tsrs/linux-x64-gnu/tsrs

# in a project directory
$GO71 -p . --noEmit --checkers 4 --extendedDiagnostics
$RS   -p . --noEmit --checkers 4 --extendedDiagnostics
$RS   -p . --noEmit --checkers 4 --checkerAssignment go --noLazyMembers
```

If you check a project with `composite: true` (like TypeScript's own `src/`), add
`--composite false --incremental false`. Otherwise every run after the first reads a
`.tsbuildinfo` and returns in about 0.3 s. That cost me a benchmark run before I noticed.
