---
title: "Python 3.15's lazy imports, measured: 141ms to 19ms"
description: PEP 810 adds a `lazy` keyword to Python 3.15. I ran it on 3.15.0b4 against a realistic CLI tool. The common path dropped from 141ms to 19ms and from 322 modules to 60, the hot path was unchanged, the bytecode differs by a single flag bit, and a typo'd import stays silent until first use.
date: 2026-10-21
draft: false
---

Python 3.15 ships [PEP 810, explicit lazy imports](https://peps.python.org/pep-0810/):
a soft keyword `lazy` that defers loading a module until the first time you
touch the name. It's the feature people have asked for since the implicit
`-X lazy_imports` experiment was pulled from 3.7, and the Steering Council
accepted it about a month after the PEP was filed. Simon Willison
[flagged](https://simonwillison.net/2026/Oct/12/python-3-15/) the 3.15 release,
and JetBrains wrote
[an explainer](https://blog.jetbrains.com/pycharm/2026/06/explicit-lazy-imports-are-coming-to-python-3-15/).

Those are good overviews of *what* it does. This post is the part the overviews
skip: I installed `3.15.0b4` and measured it. Everything below is my own
benchmarking on a 4-core Xeon (2.1GHz), Linux 6.18, CPython `3.15.0b4`. The
numbers are medians over 15–25 fresh process launches.

## The syntax, and what it does to a name

```python
lazy import json
lazy from json import dumps
```

After `lazy import json`, the name `json` is bound but the module is not
loaded. It sits in a new `sys.lazy_modules` set instead of `sys.modules`.
The first time anything reads the name, the real import runs, the module moves
into `sys.modules`, and the name is replaced in place:

```python
import sys
lazy import json
# A. right after the statement
"json" in sys.lazy_modules   # True
"json" in sys.modules        # False
_ = json                     # a bare name load is enough to resolve it
# B. after touching the name
"json" in sys.lazy_modules   # False
"json" in sys.modules        # True
```

There is no Python-level way to inspect the binding *without* resolving it —
even `type(json)` reports `<class 'module'>` and pulls the module in. The proxy
is deliberately invisible.

## A realistic CLI, two versions

Lazy imports pay off when a program imports a heavy dependency it often doesn't
use on a given run — the classic shape of a CLI with subcommands. I built the
smallest honest version of that: a tool that needs `requests` only for its
`fetch` subcommand.

```python
# tool_eager.py
import sys
import requests                 # paid on every launch
def main():
    if sys.argv[1:2] == ["fetch"]:
        return requests.__name__
    return "help"

# tool_lazy.py
import sys
lazy import requests            # paid only if fetch runs
def main():
    if sys.argv[1:2] == ["fetch"]:
        return requests.__name__
    return "help"
```

On its own, `import requests` costs about 108ms net of interpreter startup on
this box (it drags in `urllib3`, `ssl`, `http`, `email`, and more). Here's what
that does to the two paths:

```
common path (help, requests never used)
  eager: 141.1 ms
  lazy :  18.6 ms      ~7.6x faster, ~122 ms saved

hot path (fetch, requests IS used)
  eager: 139.0 ms
  lazy : 137.5 ms      difference within measurement noise
```

That's the whole trade in two lines. The path that doesn't need the dependency
stops paying for it, and the path that does pay gains nothing and loses nothing
— the proxy resolution on first use is free relative to the import it triggers.

It isn't only wall-clock. On the `help` path I counted loaded modules and peak
RSS of the child process:

```
          modules in sys.modules   peak RSS
  eager            322               31.6 MB
  lazy              60               15.5 MB
```

262 modules and half the resident memory, gone, because the dependency graph
behind `requests` never materializes. For a short-lived CLI that mostly prints
help or errors, this is the difference between feeling instant and feeling
sluggish.

## The bytecode difference is one bit

I expected a new opcode. There isn't one. `lazy import json` compiles to the
*same* `IMPORT_NAME` instruction as a normal import — only the oparg flag
changes:

```
eager:  IMPORT_NAME  0 (json)
lazy :  IMPORT_NAME  1 (json + lazy)
```

The laziness rides along as a flag on the existing import opcode, with the
proxy machinery living in the import system rather than in a dedicated
instruction. It's a small, surgical change to the compiler surface for a fairly
large change in semantics.

## CPython is already eating its own dog food

Start a clean 3.15 interpreter and look at `sys.lazy_modules`:

```python
$ python3.15 -c "import sys; print(sorted(sys.lazy_modules))"
['locale', 'pkgutil', 'traceback', 'warnings']
```

Nobody wrote `lazy import` in your program yet, but four standard-library
modules are already deferred. CPython's own startup has been converted to use
the feature internally: `traceback`, `warnings` and friends only get built if
something actually needs them. The feature isn't just offered to you; the
interpreter spends it on its own cold start.

## Two sharp edges

The convenience has semantics you have to respect, and they bite in ways a
"7x faster startup" headline won't warn you about.

**A typo is silent until first use.** A lazy import of a module that doesn't
exist succeeds:

```python
lazy import totally_not_a_real_module_xyz   # no error here
...
totally_not_a_real_module_xyz.foo           # ModuleNotFoundError, now
```

The `ModuleNotFoundError` surfaces only when some code path finally reads the
name — which might be a rare branch, in production, weeks later. Eager imports
fail at startup, loudly and on every run; lazy imports move that failure to
wherever the first use happens to be. Your import line is no longer a
smoke test.

**Top-level side effects move.** A module's top-level code — the stuff that
registers codecs, filters warnings, or monkeypatches on import — no longer runs
where you wrote the import. It runs at first use:

```
1. before lazy import
2. after lazy import statement       <- module body has NOT run
3. about to touch noisy.VALUE ...
>>> noisy.py top-level code running  <- it runs HERE, on first access
4. got 42
```

For a pure, well-behaved library this is invisible. For a module whose import
*is* the point — think `import readline` to arm tab completion, or a plugin
that registers itself as a side effect — making it lazy quietly defers or
cancels the thing you imported it for. This is exactly why PEP 810 made the
behavior opt-in per import rather than a global switch: you mark lazy the
imports you've reasoned about, and leave the side-effecting ones eager.

## So: when

Reach for `lazy` on heavy, optional dependencies in code whose startup latency
you can feel — CLIs, test collection, plugin systems, anything that imports a
big graph but uses a slice of it per run. The payoff is proportional to how
much of what you import you *don't* touch on the common path, and my numbers
suggest that for a `requests`-class dependency on a cold CLI the payoff is
large: ~120ms and ~16MB per launch, for the cost of one keyword.

Leave it off imports whose top-level code you depend on, and don't scatter it
blindly — a deferred import is a deferred failure. The feature is precise on
purpose. Used that way, it's the rare performance win that costs you a word and
nothing else.

---

*Benchmarks are my own, on CPython 3.15.0b4 (Linux x86-64, 4-core Xeon),
medians over 15–25 fresh process launches; your machine's constants will
differ but the shape won't. Feature details from
[PEP 810](https://peps.python.org/pep-0810/).*
