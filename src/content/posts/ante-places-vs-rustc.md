---
title: "Ante's place-based borrow checker, held up against rustc 1.97"
description: Ante's new design claims to accept safe programs Rust rejects and to do shared, even cyclic, mutation with no runtime checks. I compiled the Rust side of every example under stable rustc, Polonius-next and legacy Polonius, then measured what the Rust workarounds actually cost. Polonius fixes one of six. The RefCell tax ranges from 3% to 4.8×, and the biggest cost is memory layout, not the checks.
date: 2026-10-21
draft: false
---

Jake Fecher [published a redesign of Ante's borrow checker](https://antelang.org/blog/safe_cyclic_mutability/)
on Monday. Ante is a low-level functional language with Rust-style ownership,
but it has always tried to relax Rust's rule that you may alias *or* mutate,
never both. The new design does two things:

1. It tracks **places** instead of lifetimes. A reference's type records the
   set of things it may point into (`ref '(x, y) I32`), not a region of code.
2. It adds **unstable places**, written `'v!`. A reference into a vector's
   buffer is `mut 'v! t`, because mutating `v` might reallocate it. A
   reference to a field stored inline is stable, because nothing you do to the
   struct can move that field. Mutating through a `'v!` reference only
   invalidates references at `'v!!`, one level deeper.

From those two ideas the post builds safe shared mutation through `Rc` and
safe cyclic structures, "with no runtime checks". The post makes the case with
examples, so I tested it. This post is about two questions it leaves open:

- **How many of the Rust examples it calls "rejected" are really permanent
  Rust limits?** Rust's next borrow checker, Polonius, is supposed to fix some
  of them.
- **What does "no runtime checks" save?** The Rust answer to shared mutation
  is `Rc<RefCell<T>>`, which checks a borrow flag at runtime. How big is that
  tax in practice?

Everything below was measured on my machine: rustc 1.97.0 stable (2026-07-07),
a 4-core Intel Xeon at 2.1 GHz, Linux, `--release`, best of 7–9 runs. I didn't
run Ante. Its side of each comparison is the post's stated behaviour.

## Part 1: which "rejected" programs does Polonius fix?

I wrote the Rust version of each example in the Ante post as its own file and
compiled it three ways. One was stable rustc. One was `-Zpolonius=next`, the
location-insensitive "Polonius alpha" the compiler team is working to
stabilise. The last was `-Zpolonius`, the original datalog Polonius, which is
more precise but too slow to ship. The two nightly flags work on stable via
`RUSTC_BOOTSTRAP=1`.

| Example (from the Ante post) | Should it be safe? | stable | polonius=next | legacy polonius |
|---|---|---|---|---|
| Reassign a tuple while `&tuple.0` is live, then read it | yes (inline field) | E0506 | E0506 | E0506 |
| Same with `(String, String)`, reassign `.0` | yes (inline field) | E0506 | E0506 | E0506 |
| `first_mut()` + `last_mut()`, write first, read last | yes | E0499 | E0499 | E0499 |
| Iterate `ctx.entities` mutably, call `ctx.on_death(e)` | yes (touches other fields) | E0499 | E0499 | E0499 |
| `get_default`: `get_mut`, else `insert` and `get_mut` | yes | E0499 | **OK** | **OK** |
| Loop: `temp = temp.maybe_next()` | yes | E0499 | E0499 | **OK** |
| `let e = vec.get(1); vec.clear(); use e` | **no** (dangling) | E0502 | E0502 | E0502 |

The last row is the control. All three checkers correctly reject the dangling
reference.

Here is my reading of the table. Polonius is a better *lifetime* analysis. It
fixes the cases where Rust is wrong about *when* a borrow ends: a borrow
returned out of one match arm (`get_default`) or reassigned across loop
iterations (`maybe_next`). Even then, the version that's on track to ship
only gets the first of those. The loop case,
[NLL problem case #4](https://github.com/rust-lang/rfcs/blob/master/text/2094-nll.md),
still needs the slow location-sensitive analysis.

The other four rejections aren't about *when* a borrow ends. They're about
*what* a mutation can break, and no lifetime analysis can see that.
`pair.0 = "Foo".to_string()` overwrites a `String` in place. Rust's `&pair.0`
still points at a valid `String` afterwards, just a different one. Rust has no
way to say "this borrow survives writes to its target, but not
reallocations". That is exactly what Ante's `!` marker encodes. Four of the
six "should be safe" programs are outside what Polonius is even trying to do.

One caveat on the first two rows, from my own analysis rather than the post.
Rust's `&T` also promises the value *won't change* while you hold it, and
plenty of code relies on that. A language that allows rows 1–2 trades that
guarantee away. It reads naturally in Ante, where `ref` is documented as
`&Cell<T>`-like, but it isn't a free win.

## Part 2: what does skipping the runtime check buy?

The Ante post's headline claim is safe mutation through `Rc`, including
cycles, "with no runtime checks". The safe Rust equivalent is
`Rc<RefCell<T>>`. `RefCell::borrow_mut` reads a flag, panics if it's
already borrowed, sets it, and clears it when the guard drops. A statically
checked language can compile the same code down to what Rust would write
with `UnsafeCell`. So the gap between `RefCell` and `UnsafeCell` is the most
Ante's design could save. I measured that gap in two workloads.

### Workload A: chasing pointers around a cycle

A ring of `n` nodes, linked in shuffled order so each step is a dependent
load. Each step does `node.val += 1; cur = node.next`. I ran four variants:

```rust
// A: idiomatic safe Rust. Clone the Rc to move to the next node.
let next = { let mut n = cur.borrow_mut(); n.val += 1; n.next.clone().unwrap() };
// B: same RefCell nodes, but move with Rc::as_ptr (flag check, no refcount traffic)
// C: Rc<UnsafeCell<Node>>, i.e. what a static checker can emit
// D: Vec<Node> arena with usize "pointers" (the usual "just use indices" answer)
```

20M steps, ns per step:

| ring size | A: RefCell + clone | B: RefCell, no clone | C: UnsafeCell | D: indices |
|---|---|---|---|---|
| 1,000 (in L1/L2) | 2.37 | 2.04 | 1.99 | 2.14 |
| 100,000 | 29.4 | 32.9 | 32.8 | **12.9** |
| 4,000,000 | 162 | 157 | 160 | 137 |

When the ring fits in cache, the borrow flag alone (B vs C) costs about 3%.
The idiomatic `Rc::clone` per step, which increments one refcount and
decrements another, costs about 19%. Once the ring outgrows cache, A, B
and C are within noise of each other, because every step waits on a cache
miss and the flag check runs in that miss's shadow.

The real outlier is **D, the index arena, at 2.5× faster at 100k nodes**. It
has nothing to do with checks. It comes down to node size:

```
Rc<RefCell<Node>>  : strong 8 + weak 8 + borrow flag 8 + val 8 + next 8 = 40 B
Rc<UnsafeCell<Node>>: strong 8 + weak 8 +                 val 8 + next 8 = 32 B
Vec<Node> (indices) :                                     val 8 + next 8 = 16 B
```

At 100k nodes the arena is 1.6 MB and the `Rc` versions are 3.2–4 MB, plus
allocator overhead. On this machine that's the difference between staying in
L2/L3 and not. Removing the `RefCell` flag takes 8 of the 24 extra bytes. The
`Rc` header takes the other 16, and Ante keeps it.

### Workload B: touching many cells without chasing pointers

Pointer chasing hides per-access costs behind latency. The opposite case is a
flat loop over 4,096 cells (L1-resident), each doing `+= 1`, repeated 20,000
times:

| layout | ns / element |
|---|---|
| `Vec<RefCell<i64>>`, `*c.borrow_mut() += 1` | 0.58 |
| `Vec<Cell<i64>>`, `c.set(c.get() + 1)` | **0.12** |
| `Vec<UnsafeCell<i64>>` | 0.12–0.14 |
| `Vec<Rc<RefCell<i64>>>` | 1.08–1.11 |
| `Vec<Rc<UnsafeCell<i64>>>` | 0.74 |

Here the check really costs something: **4.8×** for inline cells. My
explanation, which I didn't verify against the assembly, is that each
`borrow_mut` has a possible panic. That control flow stops LLVM from
vectorising the loop, while the `Cell` and `UnsafeCell` loops run at about
four elements per cycle. Behind an `Rc`, the gap shrinks to about 1.5×
because the indirection already rules out vectorising.

Rust already has a zero-cost answer for this case, and it is `Cell`. Ante's
design generalises the idea: `Cell` allows shared mutation only when you never
hand out an interior reference, while Ante allows interior references and
tracks which mutations invalidate them. That covers the case `Cell` can't:
a `String`, a `Vec` or a whole struct behind shared mutation, not just `Copy`
scalars.

I also checked the other direction. Ante's `mut` may alias, like `&Cell`, so
the compiler loses Rust's `noalias` guarantee. For a simple
`dst[i] += src[i] * k` loop, `&mut`/`&` and `&Cell` came out identical
(0.52–0.53 ns/elem). LLVM's runtime overlap check covered the difference.
It's one data point, not proof that aliasing information never matters.

## Part 3: what the place system will cost you instead

The checks don't disappear. They move into the types, and two consequences
follow from the post's own definitions. This part is my analysis, not
something the post states.

**One place per `Rc` type parameter is coarse.** The post's cyclic list is
`type List 'p t = Nil | Cons t (Rc 'p (List 'p t))`. Every node in the list
shares the single place `'p`. So clearing or reallocating anything reached
through *any* node invalidates every unstable reference into *every* node of
that list. Stable references, such as one to an inline `I32` payload, survive.
A reference into a node's `Vec` or `String` doesn't. Fine-grained per-node
invalidation is what `RefCell` gives you at runtime, and a static place system
can't give you that for an unbounded structure. For scalar payloads this is
irrelevant. For a graph of nodes that each own a buffer, expect the checker to
make you re-fetch references after mutations more often than seems necessary.

**Signatures carry more.** `get_default` in Ante is
`mut 'r HashMap k v -> mut 'r! v can Mutate 'r, Panic`. The `Mutate 'p`
effect is what makes `a_and_b_cant_alias c c` an error at the call site. It is
also a new piece of API surface. The post itself notes that exposing all of
`'ctx` as mutable, when only `'ctx.events` is touched, invalidates references
a narrower annotation would have kept. Rust's equivalent workaround, splitting
the borrow with `let Context { entities, events, score } = ctx;`, is
zero-cost but changes the function's shape. Ante changes the signature
instead. Neither is free.

## So what's the actual win?

From these numbers, the honest summary is:

- On **expressiveness**, the win is real and mostly out of Polonius's reach.
  Two of the six safe programs become legal Rust with Polonius, and only one
  of those with the version likely to ship. The other four need a notion of
  shape stability that Rust doesn't have.
- On **performance**, removing the borrow flag is worth roughly nothing
  (3%) to modest (1.5×) in pointer-heavy code. It's worth a lot (4.8×) in
  flat loops, but there Rust already has `Cell` and indices. The larger costs
  in cyclic code come from the `Rc` header and refcount traffic, and Ante's
  design keeps `Rc` and still leaks cycles.

So the best case for places isn't "faster than `RefCell`". It's being able to
write the observer pattern, back-pointers and `get_default` in the obvious way
while the compiler still catches `vec.get(1); vec.clear()`. That's a
reasonable thing to want. It's also a different pitch from "zero-cost".

*Sources: Jake Fecher,
[Borrow Checking Zero-Cost Cyclic Mutability](https://antelang.org/blog/safe_cyclic_mutability/)
(the Ante examples and their claimed behaviour);
[RFC 2094 (NLL)](https://github.com/rust-lang/rfcs/blob/master/text/2094-nll.md)
for problem case #4. All Rust compile results and timings are my own runs, as
described above.*
