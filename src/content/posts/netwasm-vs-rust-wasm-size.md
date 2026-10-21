---
title: "C# beats Rust on Wasm size? Taking apart the 53 KB"
description: NetWasm's "hello 42" component is 10.7% smaller than Rust's. I rebuilt both and took the Rust one apart. The gap comes from panic machinery, dlmalloc, a wide WASI import list and a missing wasm-opt pass. Rust the language barely shows up.
date: 2026-10-21
draft: false
---

[NetWasm](https://github.com/zion-sati/NetWasm) is on the Hacker News front page with a
headline that sounds backwards: C# now beats Rust on WebAssembly size. Its README compares a
standalone WASI Preview 2 component that prints `42`. The NetWasm build is **47,895 bytes**
and the Rust 1.95 build is **53,635**, so C# is 10.7% smaller raw and 19.5% smaller after
Brotli-11. To its credit, the README says this is one workload, not a ranking of languages.

A program that prints two digits shouldn't take 53 KB in any language. So the useful question
isn't which language wins. It's what those 53 KB are made of, and how much of it is really
Rust. I rebuilt both sides and then pulled the Rust binary apart one layer at a time.
Everything below is my own measurement unless I say otherwise.

## Setup

- **Rust:** 1.97.0 stable, plus nightly 2026-10-20 for two variants that need it. Target
  `wasm32-wasip2`, so rustc's `wasm-component-ld` emits a component directly.
- **Rust recipe:** copied from NetWasm's
  [methodology doc](https://github.com/zion-sati/NetWasm/blob/main/docs/size-and-methodology.md).
  That's `println!("{}", 42)` with a `small` profile: `opt-level = "z"`, `lto = true`,
  `codegen-units = 1`, `panic = "abort"`, `strip = "symbols"`.
- **C#:** .NET SDK 10.0.402 and `NetWasm.Templates` 0.10.0, built with
  `dotnet publish -c Release -p:NetWasmOptimization=Oz`, which is the doc's recipe.
- **Tools:** `wasm-tools` 1.262 for section sizes, `twiggy` 0.8 for per-function sizes,
  binaryen `wasm-opt` v132, and Brotli quality 11.
- **Check:** every Rust variant below was run with `jco run` (jco 1.39) and printed `42`.

Reproduction went cleanly. NetWasm came out at **47,895 bytes, byte-for-byte the README's
number**. Rust 1.97 gave 54,856 and nightly gave 53,397, both close to the README's 53,635 for
1.95. So the comparison is real and stable across toolchains. What's left is explaining it.

## The ladder

Each row is a complete WASI P2 command component that prints `42`:

| Build | Raw bytes | Brotli-11 | vs NetWasm (raw) |
|---|---:|---:|---:|
| NetWasm 0.10, Compact GC (rebuilt here) | 47,895 | 13,660 | — |
| Rust, the doc's recipe, `println!("{}", 42)` | 54,856 | 17,301 | +14.5% |
| same, `println!("42")` (no integer formatting) | 54,856 | 17,301 | +14.5% |
| same, nightly + `-Zbuild-std` (still `panic=abort`) | 57,311 | 18,411 | +19.7% |
| `stdout().write_all(b"42\n")` instead of `println!` | 46,202 | 14,293 | −3.5% |
| doc's recipe + `wasm-opt -Oz` on the core module | 44,275 | 15,266 | −7.6% |
| `write_all` + `wasm-opt -Oz` (all stable) | 36,402 | 12,366 | −24.0% |
| `println!`, `build-std` + `-Cpanic=immediate-abort` (nightly) | 33,960 | 10,747 | −29.1% |
| same + `wasm-opt -Oz` | 30,082 | 9,867 | −37.2% |
| `#![no_std]` + `wasi` crate + tiny bump allocator | **2,901** | **933** | −93.9% |

Three things stand out.

**Integer formatting is free.** `println!("{}", 42)` and `println!("42")` produce identical
files. The formatting machinery gets pulled in anyway, so the call site makes no difference.

**Two stable-toolchain changes flip the result.** Swapping `println!` for `write_all` and
running binaryen over the output gives 36,402 bytes. That's 24% smaller than NetWasm raw and
9.5% smaller with Brotli. Neither change needs nightly or a hand-written runtime.

**The floor is about 3 KB.** That's the `no_std` row. It's only a fair comparison if you also
give up std, which most programs can't. Still, it shows how much of the 54 KB is runtime
rather than program.

## Where the 54,856 bytes go

`wasm-tools objdump` splits a component into its embedded core modules and the component-model
glue around them. `twiggy` then gives per-function sizes inside the core module. I built a
second copy of each variant with function names left in so twiggy could label things.

```
Rust, doc's recipe: 54,856 bytes
├─ component wrapper (types, imports, aliases, shims) .......  7,556
└─ main core module ......................................... 47,300
   ├─ panic + backtrace + io::Error Display + most fmt ..... ~20,900  (removed by immediate-abort)
   ├─ dlmalloc (wasi-libc) ................................. ~12,000
   ├─ bytes wasm-opt -Oz removes that LLVM -Oz left in ......  ~4,400
   └─ the rest: wasi-libc start-up and stdio, descriptor
      table, std stdout, BufWriter, main ................... ~10,000
```

The panic number is the difference between the 54,856 and 33,960 rows. The +3,914 that
`build-std` alone added on nightly suggests the true panic cost is a little higher. The
dlmalloc number is the sum of `dlmalloc`, `dlfree`, `dispose_chunk`, `realloc`,
`prepend_alloc` and `internal_memalign` in the immediate-abort build. The wasm-opt number is
26,925 → 22,543 on that build's core module. On the doc's recipe, wasm-opt removes 8,700 bytes
from the core, but some of that overlaps with the panic code.

Here's each layer in more detail.

### 1. Panic machinery (~21 KB, about 40% of the file)

`panic = "abort"` in Cargo still links std's panic *handler*. That handler formats a message,
looks up `RUST_BACKTRACE`, takes a lock and writes to stderr. Writing the message needs
`core::fmt`, and the handler needs `Display` for `io::Error`. In the unstripped build,
`panic_with_hook` alone is 2,258 bytes. `get_backtrace_style` and the `DisplayBacktrace`
formatter are another 1,155.

`-Cpanic=immediate-abort` replaces all of that with a trap. It's nightly-only and needs std
rebuilt with `-Zbuild-std`. Then `println!` and `write_all` end up 383 bytes apart, which
confirms the earlier point: fmt was never the expensive part. The panic path was.

### 2. A general-purpose malloc (~12 KB)

On `wasm32-wasip2`, Rust's std allocator sits on top of wasi-libc's `dlmalloc`. It's about
12 KB of code for a program that barely allocates anything. NetWasm offers a bump
collector, and its README shows that saves 4,031 bytes over the default Compact GC. Rust can do
the same with `#[global_allocator]`. But with std, wasi-libc still references `malloc`
internally, so you only lose dlmalloc if you leave std, as in the last row.

### 3. Nobody ran binaryen (~4–9 KB)

This is the most surprising finding. NetWasm's core module barely changes under
`wasm-opt -Oz`: 46,481 → 46,167, which is 0.7%. Either NetWasm already runs a binaryen-grade
pass or its codegen is just that tight. Rust's output from `cargo build` with `opt-level = "z"`
loses 18% of its core module to the same pass. rustc doesn't run wasm-opt itself; `wasm-pack`
and `trunk` do, but a plain `cargo build --target wasm32-wasip2` doesn't. So part of the
headline compares an optimized artifact with one that never got a post-link pass. Getting this
back takes three commands:

```sh
RUSTFLAGS="-Clink-arg=--skip-wit-component" \
  cargo build --profile small --target wasm32-wasip2      # core module, with component-type section
wasm-opt -Oz --enable-bulk-memory --enable-sign-ext --enable-nontrapping-float-to-int \
  --enable-mutable-globals --enable-multivalue --enable-reference-types \
  target/wasm32-wasip2/small/c42.wasm -o core.oz.wasm
wasm-tools component new core.oz.wasm -o c42.wasm         # re-wrap as a component
```

### 4. The import list is a size cost too (~6 KB)

This layer doesn't show up if you only look at code size. The *component wrapper* holds the
interface types, imports, aliases and canonical-ABI shims. NetWasm's wrapper is **1,414 bytes**
and Rust's is **7,556**. Both are WASI P2 commands, but they ask the host for very different
things:

```
NetWasm imports (4):  wasi:cli/environment, wasi:io/error, wasi:io/streams, wasi:cli/stdout
Rust std imports (14): the above plus wasi:io/poll, wasi:clocks/monotonic-clock,
                       wasi:cli/{stdin, stderr, exit, terminal-input, terminal-output,
                                 terminal-stdin, terminal-stdout, terminal-stderr}
```

wasi-libc's start-up sets up all three stdio streams and checks whether each one is a
terminal, whether or not the program uses them. Each imported interface brings its WIT types
into the component. The five `terminal-*` imports turn out to be dead code: `wasm-opt` deletes
them, and the re-wrapped component drops from 14 imports to 9. The wrapper shrinks to 5,675
bytes. NetWasm's closed-world compiler avoids the problem from the start by importing only what
the program reaches. That's a real design advantage, but it's a property of the runtime and
libc layer, not of C# versus Rust.

## One thing the size table doesn't show

I couldn't run the NetWasm component. jco 1.39 refused to transpile it with "exceptions
proposal not enabled". The core module has a `tags` section and over a hundred `try_table` /
`throw` instructions, so NetWasm relies on Wasm exception handling. That's fine on recent
Wasmtime, V8 and SpiderMonkey, but it narrows the set of hosts that can run it. Rust with
`panic=abort` uses no exception-handling instructions, and every Rust build above ran in jco
as-is. The baseline features each binary needs belong in a size comparison too.

## What I'd conclude

- The README's number is accurate and reproducible. NetWasm's 47.9 KB is a strong result for
  a GC'd language, and the 256× gap to Blazor in its README is the more impressive number.
- The 10.7% gap to Rust comes almost entirely from **runtime defaults**: std's panic handler,
  dlmalloc, wasi-libc's eager stdio setup, and the lack of a wasm-opt pass in plain
  `cargo build`. None of these depend on the language, and the first two are well-known
  "small Wasm in Rust" tips. Fix two of them on stable Rust and Rust ends up 24% smaller.
- Single-binary size contests mostly measure the default runtime around a two-line program.
  For a real application the better questions are how fast size grows with program size, and
  what each binary needs from its host. A `hello 42` can't answer either one.

My measurements cover one tiny program and say nothing about how either toolchain scales. I
couldn't execute the C# artifact, so its behaviour is taken from the project's own claims.

*Sources: NetWasm's [README](https://github.com/zion-sati/NetWasm) and
[size methodology](https://github.com/zion-sati/NetWasm/blob/main/docs/size-and-methodology.md).
All builds, sizes and decompositions above are
my own measurements.*
