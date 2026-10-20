---
title: "tsrs vs tsgo: how much of the speedup is Rust?"
description: I benchmarked tsrs, the new Rust port of the TypeScript compiler, against TypeScript 6 and 7 on two codebases and switched its optimizations off one at a time. On a type-heavy project it was 10× faster than the Go compiler on 4 cores. One core accounts for about 3× of that. Most of the rest comes from parallel scaling the Go build never got.
date: 2026-10-20
draft: false
---

Cloudflare and VoidZero
[announced tsrs](https://blog.cloudflare.com/typescript-rust-port/) today. It's a Rust
port of the TypeScript compiler, built inside the [Oxc](https://github.com/oxc-project/tsrs)
project and synced with upstream by agents. The pitch is TypeScript 7's Go compiler
("tsgo"), only faster and leaner: the announcement claims up to 8× faster than
TypeScript 7.0 at the same number of type-check workers, with about 70% less memory.

Claims like "Rust port is 8× faster than Go port" invite a lazy reading: Rust beats Go.
But a port is more than a translation. tsrs's own engineering notes say the maintainers
dropped the rule of mirroring tsgo function for function, as long as results stay the
same. So the speedup is a mix of language, memory management, algorithm changes and
scheduling. I wanted to know how much each one contributes, and whether "drop-in" holds
up. tsrs exposes flags that turn some of its divergences off, so this can be measured
rather than guessed.

Everything below is my own measurement unless I attribute it. Setup: a 4-vCPU Intel Xeon
VM (2.1 GHz, 15 GB RAM), Linux, `tsrs 0.18.0-ts7.1.0-dev.20260929`. Timings are medians of
5 runs after a warmup (3 for the ablations), and runs vary by roughly ±10%.

## The contestants and the workloads

tsrs pins an upstream commit and reports it in `--version`, so I could install the
TypeScript nightly built from the same day. That gives a fair Go baseline: same language
semantics, same diagnostics, different implementation.

| label | what it is |
|---|---|
| ts6 | `typescript@6.0.3`, the last JavaScript compiler |
| ts7.0 | `typescript@7.0.2`, the Go compiler, current `latest` |
| ts7.1β | `typescript@7.1.0-beta`, Go, which the announcement says carries some optimizations upstreamed from tsrs |
| go-0930 | `typescript@7.1.0-dev.20260930.4`, the Go nightly matching tsrs's upstream pin |
| tsrs | `tsrs@0.18.0-ts7.1.0-dev.20260929`, native linux-x64 binary |

I called every native binary directly to skip the Node launcher. Two workloads:

- **TS repo**: the TypeScript 6.0.3 source tree (`src/**`, 950 files, 576k lines including
  libs), flattened into one `--noEmit` project. This is ordinary application-style code.
  On this config every compiler reports the same 24 errors, from lib mismatches in my
  setup. That's useful, because it means there are diagnostics to compare.
- **type-fest**: [sindresorhus/type-fest](https://github.com/sindresorhus/type-fest) at HEAD
  (`9c943a5`), checked with its own `tsconfig.json`. It has 52k lines of conditional,
  mapped and template-literal types plus their type tests. This is the worst case for a
  type checker, and it compiles clean.

## Headline numbers

| | TS repo wall | TS repo RSS | type-fest wall | type-fest RSS |
|---|---|---|---|---|
| ts6 (JS) | 22.50 s | 1,065 MB | 102.9 s | 7,229 MB |
| ts7.0 (Go) | 3.84 s | 969 MB | 31.2 s | 9,173 MB |
| ts7.1β (Go) | 2.62 s | 711 MB | 20.8 s | 6,387 MB |
| go-0930 (Go) | 2.96 s | 886 MB | 25.2 s | 8,752 MB |
| **tsrs** | **0.71 s** | **316 MB** | **2.52 s** | **1,538 MB** |

Against the matched Go nightly, tsrs is **4.2× faster on the TS repo and 10.0× on
type-fest**, with 2.8× and 5.7× less peak memory. Against the newer 7.1 beta the gap is
3.7× and 8.3×. On type-fest, tsrs takes 2.5 s where TypeScript 6 takes 103 s.

All five compilers produced the same diagnostics on both projects (the error lines hash
identically after sorting), and the result was stable across every run.

So the claim holds up on my machine. The gap also depends heavily on the workload: 4× on
plain code, 10× on type gymnastics. Next, where does it come from?

## Taking the 10× apart

Three knobs isolate the parts:

- `--singleThreaded` (both compilers) removes parallelism. What's left is per-core
  efficiency.
- `--noLazyMembers` (tsrs) turns off tsrs's lazy member tables. As far as I can tell from
  the counters below, this is the biggest algorithmic divergence from tsgo.
- `--checkerAssignment go` (tsrs) makes tsrs split files across checker threads the way
  tsgo does.

type-fest, wall-clock medians:

| config | Go (go-0930) | tsrs | tsrs, no lazy | tsrs, Go assignment | tsrs, Go assignment + no lazy |
|---|---|---|---|---|---|
| 1 thread | 33.7 s | 9.25 s | 10.40 s | n/a | n/a |
| 4 checkers | 25.2 s | 2.52 s | 2.93 s | 4.90 s | 5.52 s |
| CPU time, 1 thread | 43.9 s | 9.2 s | | | |
| CPU time, 4 checkers | 74.9 s | 9.5 s | 10.9 s | 9.5 s | 10.2 s |

The 10× factors cleanly (my arithmetic on the medians above):

```
25.2 / 2.52  =  10.0×  total

  33.7 / 10.40  = 3.2×   one core, lazy members off  ("the port itself")
  10.40 / 9.25  = 1.1×   lazy member tables
  (9.25/2.52) / (33.7/25.2)
     = 3.7 / 1.3 = 2.7×   tsrs gets 3.7× from 4 threads, Go gets 1.3×

  3.2 × 1.1 × 2.7 ≈ 10
```

For the TS repo the same chain gives 2.8× single-core, about 1.0× from lazy members (2.00 s vs
2.01 s over 9 runs; it only saves 4% memory here), and 1.6× from scaling (tsrs 2.9× on 4
threads, Go 1.9×), for about 4.4× total. That run set gave 4.4× rather than the 4.2× in the main table.

**The language-and-implementation part is about 3×, on both workloads.** That covers
Rust, arena allocation, data layout, and tsrs's extra checker memos (inference, union,
indexed-access and flow caches), which per tsrs's docs only `--checkerAssignment go` turns off. The rest of the headline number depends on
the workload, and on type-fest it's mostly parallel scaling.

### Are they doing the same work?

Faster could also mean doing less. `--extendedDiagnostics` prints counters, so I compared
them with one checker:

| type-fest, 1 checker | Go | tsrs | tsrs, no lazy |
|---|---|---|---|
| Instantiations | 18,059,974 | 18,026,279 | 18,059,075 |
| Symbols | 16,201,629 | 7,565,955 | 16,201,629 |
| Check time | 40.1 s | 8.1 s | 11.2 s |
| "Memory used" | 4.96 GB | 1.16 GB | 1.84 GB |

Instantiations agree to within 0.2%. tsrs isn't skipping type-level work. It does the
same instantiations, and with lazy members off it gets through them 3.6× faster (40.1 s vs 11.2 s). In that mode the
symbol count matches tsgo **exactly**, which is good evidence that the flag reproduces
tsgo's member resolution. With lazy members on, tsrs builds fewer than half the symbols.
The name suggests that members of instantiated and mapped types are created on demand rather than all at once; I haven't read the code to confirm it.
That saves 36% of memory on type-fest and about 27% of single-thread check time. On the
TS repo, which has few mapped types, it does nothing measurable.

One thing I can't explain: the "Types" counter is 5.94M for Go and 1.33M for tsrs, with or
without lazy members, while instantiations match. My guess is that the two implementations
don't count the same thing, so I'm not reading anything into it.

### Why Go doesn't scale on type-fest

This was the surprise. With 4 checkers, tsgo spends **71% more CPU** than with 1 (74.9 s
vs 43.9 s) and finishes only 1.3× sooner. tsrs's CPU barely moves (9.2 → 9.5 s).

Duplicated work isn't the explanation. Each checker re-derives types that files share,
but instantiations grow about the same in both compilers (Go 18.06M → 20.54M, tsrs
18.03M → 20.46M).

Two other things show up in the measurements:

1. **Load balance.** Forcing tsrs onto tsgo's file assignment nearly doubles its 4-thread
   wall time (2.52 → 4.90 s) at identical CPU time. So that's idle threads, not extra
   work. type-fest's test files vary a lot in size (`paths.ts` alone is 60 KB of nested
   type tests). tsrs's notes describe work stealing between checkers, which handles that
   imbalance. Under the same assignment, tsrs still scales 1.9× and Go 1.3×.
2. **The kernel.** I took one sample with `getrusage`, so treat these as indicative:

   | type-fest | user | sys | minor page faults | peak RSS |
   |---|---|---|---|---|
   | Go, 1 checker | 36.0 s | 5.0 s | 1.76M | 5.9 GB |
   | Go, 4 checkers | 50.2 s | **22.0 s** | 2.89M | 8.9 GB |
   | tsrs, 1 checker | 7.9 s | 3.2 s | 0.10M | 1.1 GB |
   | tsrs, 4 checkers | 9.6 s | 1.3 s | 0.13M | 1.5 GB |

   tsgo spends about a third of its 4-checker CPU in the kernel, faulting in fresh heap
   pages on 4 threads at once. tsrs takes 20× fewer page faults. Its notes say it opts
   into transparent huge pages, which on Linux turns 512 faults into one, and this VM has
   THP set to `madvise`, so only programs that ask get huge pages. I tried to see whether
   this is Go's garbage collector: `GOGC=off GOMEMLIMIT=12GiB` cut Go's user time, but the
   4-checker run still spent 31 s in sys and took 33 s wall. So the cost is in fresh
   memory, not in collection. I couldn't flip the system THP setting in this sandbox,
   so I haven't tested whether `always` would close the gap. If you run tsgo in CI on
   big type-level code, that's worth trying.

So most of the 10× isn't Rust being faster than Go at type checking. Per core, it's 3×.
The rest is the memory model: arenas and huge pages instead of a GC heap faulted in page
by page, and that matters more as threads compete for the kernel. It's also scheduling,
which isn't a language feature and which upstream could adopt. The announcement says
some tsrs optimizations already went upstream, and the 7.1 beta is 21% faster than the
matched nightly on type-fest.

## Is it really drop-in?

The tsrs README claims error baselines match on all but 4 of 13,462 conformance tests,
and that the authors haven't found a correctness bug on real projects. I ran a check of
my own. I sampled 3,000 test files from TypeScript's `tests/cases/compiler` and
`tests/cases/conformance`, ran each one alone with my own options
(`--strict --target esnext --lib esnext,dom`, ignoring the harness directives), and
diffed the full output of tsrs against the matched Go nightly:

```
files compared        3000
files with errors     1991
diagnostics           9428
byte-identical output 3000   (including order and exit code)
```

This isn't fully independent. tsrs's own suite runs these same files, though with each
test's own options, not mine. But perfect agreement under options neither suite uses,
plus the two real projects, is a strong signal for `--noEmit` checking. I didn't test
emit, `--build`, watch mode or the language server. By tsrs's own numbers, that's where
it's less complete: 4,074 of 4,546 fourslash groups pass.

## What I'd take from this

- **If you type-check in CI, tsrs is worth trying now.** On type-heavy code it's an order
  of magnitude faster and uses about 1/6 the memory, with identical diagnostics in
  everything I threw at it. On ordinary code, expect around 4×. The gap over TypeScript
  7.0, which is what most people would switch from, is bigger still.
- **The benefit grows with core count.** tsrs's own published table compares tsgo at its
  default 4 checkers against tsrs at 16. That's a real default-vs-default comparison, but
  on a 16-core box it rewards tsrs's scaling as much as its code. On a 4-core runner like
  mine, the per-core factor is about 3× and the scaling factor is 1.6–2.7×, depending on
  the workload.
- **"Rust vs Go" is the least interesting part of the result.** The per-core gap is real
  but moderate. The large factors come from memory management and scheduling, and some
  of the scheduling and laziness wins could in principle move upstream into Go.

## Reproduce it

The harness is a fork/`wait4` loop that records wall time, CPU time and `ru_maxrss` per
run. The compilers were invoked like this (paths shortened):

```sh
npm i tsrs@0.18.0-ts7.1.0-dev.20260929 ts6@npm:typescript@6.0.3 ts70@npm:typescript@7.0.2 \
      ts71@npm:typescript@7.1.0-beta ts71dev@npm:typescript@7.1.0-dev.20260930.4

# native binaries, no Node launcher
GO=node_modules/ts71dev/node_modules/@typescript/typescript-linux-x64/lib/tsc
RS=node_modules/@tsrs/linux-x64-gnu/tsrs

$GO -p tsconfig.json --pretty false [--singleThreaded | --checkers N] --extendedDiagnostics
$RS -p tsconfig.json --pretty false [--singleThreaded | --checkers N] \
    [--noLazyMembers] [--checkerAssignment go] --extendedDiagnostics
```

For the TS repo, run `npx hereby generate-diagnostics` first, then use a
`src/tsconfig.bench.json` that extends `tsconfig-base` with `composite`, `declaration`,
`isolatedDeclarations` and `sourceMap` off, `noEmit` on, and `types: ["node","mocha","chai"]`.
