---
title: "Futhark's division fix, without the 2.3×: it was never the branch"
description: Futhark made INT_MIN / -1 well-defined and CUDA division got 2.3× slower. I compiled the old and new code down to NVIDIA SASS. The cost isn't the branch. It's a reciprocal that stops being shared. Moving one ternary brings the fix down to about 6%.
date: 2026-10-20
draft: false
---

Last week Troels Henriksen wrote up
[a small, honest fix in Futhark](https://futhark-lang.org/blog/2026-10-14-defining-division.html)
(it's on the lobste.rs front page today). Futhark compiles to C. In C, `INT32_MIN / -1`
overflows, which is undefined behaviour, and on x86 it traps. Futhark defines integer
overflow to wrap, so the runtime now special-cases a divisor of `-1`:

```c
int32_t squot32(int32_t x, int32_t y) {
  return y == -1 ? (int32_t)(0 - (uint32_t)x) : x / y;
}
```

His measurements: no difference on the sequential C backend, and a division-heavy
microbenchmark on CUDA went from 5.5 ms to 12.6 ms, about **2.3× slower**. He puts that
down to branches splitting up a run of `div.s32` instructions that used to sit in one basic
block. He ends by saying he'd like a cheaper fix but hasn't found one.

I went looking for one. The short version:

1. The slowdown isn't the branch. NVIDIA GPUs have no integer divide instruction. Each
   `div.s32` becomes a reciprocal-based routine, and the expensive part of that routine
   (`I2F` → `MUFU.RCP` → `F2I`) depends only on the divisor. Before the fix, `ptxas` computed
   it once and reused it for 8 unrolled divisions. After the fix it computes it 8 times.
2. The compiler couldn't turn the ternary into a select because evaluating `x / y`
   unconditionally might be the very UB we're avoiding. The fix is to make the divisor safe
   *and* move the division out of the ternary. That second part matters: Futhark's own ISPC
   backend already uses a safe divisor and still gets the full slowdown on CUDA.
3. With that change the hot loop is 121 SASS instructions instead of 243 (114 before the
   fix). Remainder ends up byte-for-byte identical to the unfixed code.

Everything below is my own measurement, except the 2.3× figure and the benchmark program,
which come from Henriksen's post. **One caveat up front: I don't have a GPU in this sandbox.**
I compiled to real SASS with NVIDIA's `ptxas` and read the machine code. I didn't time
anything on a GPU. Treat the instruction counts as a strong prediction, not a benchmark.

## The setup

Henriksen's benchmark (`divisions.fut`) runs `z = z // y` a million times per element, with
`y` fixed for each thread. I wrote the same thing as a C kernel and tried four versions of
the division:

```c
// old: what Futhark emitted before the fix (UB on INT_MIN / -1)
int32_t q_old(int32_t x, int32_t y) { return x / y; }

// fut: Futhark's current fix
int32_t q_fut(int32_t x, int32_t y) { return y == -1 ? (int32_t)(0 - (uint32_t)x) : x / y; }

// sel: make the divisor safe, divide unconditionally, fix the sign afterwards
int32_t q_sel(int32_t x, int32_t y) {
  int32_t ys = y == -1 ? 1 : y;
  int32_t q  = x / ys;                       // never INT_MIN / -1
  return y == -1 ? (int32_t)(0 - (uint32_t)q) : q;
}

// min: only rewrite the single bad pair; INT_MIN / 1 == INT_MIN, the wrapped answer
int32_t q_min(int32_t x, int32_t y) {
  int32_t ys = ((x == INT32_MIN) & (y == -1)) ? 1 : y;
  return x / ys;
}

// kernel body, per thread
int32_t z = xs[i], y = ys[i];
for (int32_t j = 0; j < k; j++) z = q_VARIANT(z, y);
out[i] = z;
```

The toolchain was clang 18 (`--target=nvptx64-nvidia-cuda -O3`) to PTX, then `ptxas` from
CUDA 12.9 to SASS for sm_80, then `nvdisasm`. Futhark's CUDA backend goes through NVRTC
rather than clang. Both are LLVM-based, and my PTX for `fut` looks like what the post
describes (a chain of branches between the divisions, where `old` has one straight run).

## What the machine code says

Both compilers unroll the loop 8×. Here's the main loop body for each variant on sm_80:

| variant | SASS instr. in loop (8 divisions) | per division | `MUFU.RCP` | branches | `BSSY`/`BSYNC` regions |
|---|---:|---:|---:|---:|---:|
| `old` (UB) | 114 | 14.3 | 1 | 1 | 0 |
| `fut` (current fix) | 243 | 30.4 | **8** | 16 | 8 |
| `sel` | 121 | 15.1 | 1 | 1 | 0 |
| `min` | 231 | 28.9 | **8** | 1 | 0 |

sm_75 and sm_90 give the same picture (old 114, fut 242–251, sel 121–122, min 231).

The ratio `fut / old` is 243 / 114 = **2.13×**. That's close to the 2.3× Henriksen measured.
The extra gap is plausibly `MUFU` throughput and the divergence bookkeeping, which cost more
than an average instruction.

To see where the instructions go, here's one division from the `fut` loop, trimmed:

```text
BSSY B0, `(.L_x_74)          ; open a divergence region (some lanes may have y == -1)
@!P1 BRA `(.L_x_75)          ; y == -1 lanes skip the division
IABS   R11, R4
I2F.RP R6, R11               ; ┐
MUFU.RCP R6, R6              ; │ reciprocal of |y|, recomputed for every division
IADD3  R8, R6, 0xffffffe, RZ ; │
F2I.FTZ.U32.TRUNC.NTZ R9, R8 ; ┘
IMAD.HI.U32 ...              ; quotient estimate and two correction steps
...
BSYNC B0                     ; reconverge
```

In `old`, the four marked instructions appear once per unrolled block, and each of the 8
divisions is just `IMAD.HI.U32` plus a few corrections. That's the real cost of the fix.
The reciprocal of a loop-invariant divisor stops being shared, because each division now
lives in its own conditional block, and `ptxas` doesn't recombine them.

`min` is a useful negative result. It has no branches at all and is still 2× slower. Its
divisor depends on `x == INT32_MIN`, and `x` changes every iteration, so the divisor is no
longer loop-invariant and the reciprocal has to be recomputed every time. The cleverest-looking
fix (one select, no negation) is the worst one on a GPU. On a GPU, what matters is whether
the divisor stays invariant, not how many instructions you save.

## Why the compiler couldn't fix this itself

`y == -1 ? A : x / y` looks like an obvious candidate for a branch-free select: compute both
sides and pick one. LLVM won't do it, because computing `x / y` on the path where the source
didn't ask for it is only legal if the division can't trap. For `sdiv`, that means the
compiler has to prove the divisor is neither 0 nor -1. It can't, so the division stays
behind a branch.

You'd think a safe divisor would be enough on its own. It isn't. Futhark's ISPC backend
(the `#if defined(ISPC)` half of
[`rts/c/scalar.h`](https://github.com/diku-dk/futhark/blob/master/rts/c/scalar.h)) already
does this:

```c
ys = y == -1 ? 1 : y;
return y == -1 ? (int32_t)(0 - (uint32_t)x) : x / ys;
```

I compiled that shape for CUDA too: **243 instructions, 8 `MUFU.RCP`, 8 divergence regions**,
the same as `fut`. `ys` can still be 0 as far as the compiler knows, so `x / ys` still can't
be evaluated where the source didn't ask for it. `sel` gets around this by not making the
compiler speculate: the division runs on every path in the source, and only the cheap negation
is conditional. That's legal to turn into a select (`selp` in PTX, `SEL` in SASS), and LLVM
even notices that two negations cancel across unrolled iterations.

## The other operators

The same change works for the rest of Futhark's signed family. Futhark defines
`INT_MIN % -1` as 0, and `x % 1` is already 0, so remainder needs no fix-up at all:

```c
int32_t srem32(int32_t x, int32_t y) { return x % (y == -1 ? 1 : y); }
```

Floor division (`sdiv32`) and floor modulus (`smod32`) work the same way: divide by `ys`,
then negate the result if `y == -1` (or return the modulus, which is 0). Same benchmark
shape, sm_80 loop bodies:

| op | unfixed (UB) | Futhark's fix | safe divisor, unconditional op |
|---|---:|---:|---:|
| `squot` (`//`) | 114 | 243 | 121 |
| `sdiv` (`/`, floor) | 93 | 147 | 98 |
| `srem` (`%%`) | 13 | 228 | **13 (identical)** |

The `srem` row is the most striking. Before the fix, LLVM used `(z % y) % y == z % y` to
fold each unrolled block of eight remainders into a single `srem` (I checked the IR). The
branch hides that from it, so all eight remainders come back, each with its own reciprocal,
and the loop body grows 17×. With the safe divisor, the generated code is the same as the unfixed version.

## Correctness

These rewrites are only useful if they're exactly right, including the case that started
all this. I checked `sel`-style quotient, remainder, floor division and floor modulus against
a 64-bit reference that wraps the result to 32 bits. The test covered every pair from a list
of edge values (`INT32_MIN`, `INT32_MIN+1`, ±1, ±2, ±3, ±7, `INT32_MAX`, …) and about
**198.6 million** random pairs, biased toward small divisors and `x = INT32_MIN`. It ran under
GCC's `-fsanitize=undefined -fno-sanitize-recover=all`. **Zero mismatches and zero UBSan
reports.** `INT32_MIN / -1` gives `INT32_MIN`, and both remainder operators give 0, as
Futhark specifies.

## And on the CPU?

Henriksen saw no difference on the C backend, and I don't either. Here's 100 chained
divisions per element over 10⁶ elements, best of 5, on a 2.1 GHz Xeon:

| compiler | divisors | old | fut | sel | min |
|---|---|---:|---:|---:|---:|
| gcc -O3 | wide range | 356 ms | 350 ms | 346 ms | 365 ms |
| clang -O3 | wide range | 340 ms | 340 ms | 343 ms | 347 ms |
| gcc -O3 | {-2..2} | 343 ms | 285 ms | 282 ms | 366 ms |
| clang -O3 | {-2..2} | 345 ms | 282 ms | 365 ms | 357 ms |

On x86, `idiv` latency dominates and a well-predicted branch costs almost nothing. When a
fifth of the divisors are -1, the branchy `fut` actually wins, because it skips the `idiv`
for those lanes. So no single formulation wins everywhere. If a code generator emits
separate CPU and GPU runtimes (Futhark already has `#ifdef`s for this), the GPU side should
use the unconditional-division form.

## What I'd take away

- **On GPUs, the cost to watch is the reciprocal setup, not the branch.** An integer
  division by a value the compiler can see is invariant is cheap. Anything that hides that
  invariance from `ptxas` costs you a `MUFU.RCP` per division: putting the division in its
  own conditional block, or making the divisor depend on the dividend.
- **"Make the divisor safe" is only half the fix.** The division also has to be
  unconditional in the source. Otherwise the compiler isn't allowed to evaluate it on every
  path, and you get branches back.
- **I haven't run this on a GPU.** If you have one and a Futhark checkout, swapping the four
  `squot`/`sdiv`/`srem`/`smod` bodies in the non-ISPC half of `rts/c/scalar.h` and re-running
  `divisions.fut` would settle it in a few minutes. My prediction is within about 10% of the
  pre-fix time.

## Reproduce it

```sh
python3 -m venv v && v/bin/pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-nvdisasm
N=v/lib/python3.*/site-packages/nvidia
clang --target=nvptx64-nvidia-cuda -march=sm_80 -O3 -S -o gk.ptx gk.c   # gk.c: kernels above
$N/cuda_nvcc/bin/ptxas -arch=sm_80 -O3 gk.ptx -o gk.cubin
$N/cu13/bin/nvdisasm -c gk.cubin | less   # look for MUFU.RCP and BSSY inside the loop
```

The C kernels need `__attribute__((nvptx_kernel))` and `__nvvm_read_ptx_sreg_tid_x()` and
friends for the thread index. Don't include `<stdint.h>` on this target; `typedef int int32_t;`
is enough.

Sources: Troels Henriksen, ["Making all divisions slower to fix a bug…"](https://futhark-lang.org/blog/2026-10-14-defining-division.html);
Futhark commit [`6cca9ee`](https://github.com/diku-dk/futhark/commit/6cca9ee3851c25edd4eaf31ce230055d80fd490d)
and [`rts/c/scalar.h`](https://github.com/diku-dk/futhark/blob/master/rts/c/scalar.h).
