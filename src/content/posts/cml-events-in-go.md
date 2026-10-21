---
title: "Concurrent ML's events in 150 lines of Go, and the part Go can't do"
description: A post on lobste.rs argues that Concurrent ML's first-class events beat Go's select, and that bolting them onto Go is unrealistic. I built them on reflect.Select anyway. Choice, wrap, guard and nack all work, at about 4.5× the cost of a native select. The part that doesn't port is reclaiming goroutines, and Go 1.27's new leak detector shows why.
date: 2026-10-21
draft: false
---

Nikhil Marathe's [ConcurrentML: Improving on Go's concurrency model](https://nikhilism.com/post/2026/golang-racket-concurrency/)
(on [lobste.rs](https://lobste.rs/) today) compares Go's channels and `select`
with Racket's implementation of Concurrent ML (CML). The argument goes like
this. In Go, a channel operation happens as soon as execution reaches it, and
a `select` case is syntax. You can't store a case, return it from a function or
combine it with another one. In CML, `channel-put-evt` returns an *event*,
which is a value that describes an operation without doing it. Events compose
with `choice-evt`, `wrap-evt`, `guard-evt` and `nack-guard-evt`, and nothing
happens until you call `sync`. The post also says that bringing this to Go is
"not realistic at all", and it notes that Go 1.27 added a goroutine leak
detector based on ideas close to CML's thread GC.

The article argues from expressiveness and has no measurements. So I tested
the "unrealistic" claim directly: I built the CML combinators in Go, ported the
article's examples, benchmarked them, and checked what Go 1.27's leak detector
actually does. Everything below is my own code and my own numbers. I used Go
1.27.0 on a 4-core 2.1 GHz Xeon running Linux.

## The short version

| Question | Answer |
|---|---|
| Can Go express first-class, composable events? | Yes. About 150 lines on `reflect.Select`, with generics for types. |
| Do nacks (abandonment notification) work? | Yes. The article's print-service example behaves the same in Go. |
| Cost of a 2-way choice | 95 ns native `select`, 150 ns `reflect.Select`, 450 ns as an event, 650 ns with a nack |
| Can Go reclaim a goroutine blocked forever? | No. Go 1.27 *detects* these goroutines but frees nothing: 200 MiB stayed allocated in my test. |
| Does that matter for CML style? | Yes. The usual nack pattern leaks one goroutine every time the guarded event *wins*. |

## How the library works

An event is a tree. The leaves are channel operations. The inner nodes are
choice, wrap (apply a function after commit), guard (run a maker function when
synced) and nack-guard (a guard whose maker also receives a "you lost"
channel):

```go
type Event[T any] struct{ n node }

func Recv[T any](ch <-chan T) Event[T]
func Send[T any](ch chan<- T, v T) Event[struct{}]
func Choose[T any](es ...Event[T]) Event[T]
func Wrap[T, U any](e Event[T], f func(T) U) Event[U]
func Guard[T any](mk func() Event[T]) Event[T]
func WithNack[T any](mk func(nack <-chan struct{}) Event[T]) Event[T]
func Sync[T any](e Event[T]) T
```

`Sync` makes one pass over the tree. Guards run their makers as they're
reached. Each nack-guard gets a fresh channel. Every leaf goes into a flat
`[]reflect.SelectCase`, and each leaf remembers two things: its chain of wrap
functions, and the nack-guards it sits under. Then:

```go
// simplified from the real Sync
i, v, ok := reflect.Select(fl.cases)   // the single commit point
committed = true
l := fl.leaves[i]
for id, ch := range fl.nacks {
    if !l.under(id) { close(ch) }      // every losing subtree learns it lost
}
r := l.conv(v, ok)
for _, f := range l.wraps { r = f(r) } // wraps run after commit, innermost first
return r.(T)
```

Two Go features make this smaller than I expected. Closing a channel is a
permanent broadcast, and that is exactly the semantics a nack needs: once it
fires, it stays ready for any number of listeners. And `reflect.Select` already
supplies the hard part of CML, which is atomically committing to one of N
channel operations with random fairness. What's left is bookkeeping.

There is one real gap. Racket's `sync` may never run a guard's maker if
another event is already ready. Mine always runs every maker. The result is
still correct, because the loser's nack fires right away, but a nack-guard that
starts a goroutine always pays for that goroutine.

## The article's print service, in Go

The article's centerpiece is a print server. A client's request should never
be enqueued if the client has already given up, and the server should never
block on a reply that no one will read. Here is my port. The public API is an
event, not a function call:

```go
func PrintEvt(msg string) Event[string] {
    return WithNack(func(nack <-chan struct{}) Event[string] {
        resp := make(chan string)
        go Sync(Choose(
            Wrap(Send(requests, req{msg, resp, nack}), ignore),
            Wrap(Recv(nack), func(struct{}) struct{} {
                fmt.Println("  client: abandoned before enqueue:", msg); return struct{}{} }),
        ))
        return Recv((<-chan string)(resp))
    })
}

// server, after doing 50ms of work:
Sync(Choose(
    Wrap(Send(r.resp, "printed "+r.msg), ...),
    Wrap(Recv(r.abandoned), /* log "client gone" */ ...),
))
```

A caller races it against a timeout with
`Sync(Choose(PrintEvt("a"), timeout(10*time.Millisecond)))`. I sent "a", then
"b" while the server was still busy, then "c" with a generous timeout:

```
1: timeout
2: timeout
  client: abandoned before enqueue: b
  service: client gone, dropping reply for a
3: printed c
```

This matches the article's Racket output case for case. Request "a" was
accepted, but its reply was dropped. Request "b" never reached the queue.
Request "c" went through. The server never blocked.

The other example in the article is a race between two HTTP fetches with an
overall timeout. The article says Go needs `context.Context` and extra
goroutines for this. With events, the nack *is* the cancellation signal, so
each fetch cancels its own context when it loses:

```go
func FetchEvt(url string) Event[string] {
    return WithNack(func(nack <-chan struct{}) Event[string] {
        ctx, cancel := context.WithCancel(context.Background())
        go func() { <-nack; cancel() }()
        out := make(chan string, 1)
        go func() { /* http GET with ctx; send body to out */ }()
        return Recv((<-chan string)(out))
    })
}

Sync(Choose(FetchEvt(a), FetchEvt(b), timeout))
```

Against a local `httptest` server, a 20 ms vs 90 ms race returned the 20 ms
body, and the 90 ms request was cancelled. A 400 ms vs 500 ms race hit the
150 ms timeout, and both requests were cancelled. The caller never sees a
context. So "Go can't compose this" is wrong at the library level. Go can't
compose it with *native* `select`, and that's a narrower claim.

## What it costs

I benchmarked with `go test -bench` and three runs each. The table shows
typical values. All the single-goroutine cases receive from a buffered channel
that is already ready, so the numbers measure select machinery, not scheduling.

| Benchmark | ns/op | allocs/op |
|---|---:|---:|
| native `select`, 2 cases | 95 | 0 |
| `reflect.Select`, 2 cases | 150 | 2 |
| `Sync(Choose(Recv a, Recv b))`, prebuilt | 450 | 6 |
| same, each branch wrapped | 510 | 8 |
| same, one branch under `WithNack` | 660 | 11 |
| ping-pong across goroutines, native | 600 | 0 |
| ping-pong across goroutines, events | 2,500 | 23 |

The event layer adds roughly 300 ns and 4 allocations on top of
`reflect.Select`, which already costs 1.5× a native select. Every `Sync`
rebuilds the flat case list, because guards are allowed to return a different
event each time. A cache for guard-free trees would remove most of that cost. I
didn't build one.

Width scales linearly, and most of the cost is Go's own:

| alternatives | `reflect.Select` | event `Sync` |
|---:|---:|---:|
| 8 | 0.79 µs | 1.9 µs |
| 64 | 10 µs | 16 µs |
| 512 | 82 µs | 140 µs |

A CPU profile of the 512-way case put 36% in `runtime.selectgo` and 17% in my
tree walk. The rest was mostly lock and GC work. Go's `selectgo` sorts and
locks every channel in the case list on each call, so a wide select costs
about 160 ns per case before any CML layer is involved.

My read is that a 4–5× overhead per operation rules this out for hot paths
like ping-pong pipelines. It's irrelevant for the use the article cares about,
which is request/response services with timeouts and abandonment, where one
network round trip costs about as much as a thousand of these syncs.

## A bug the article warned about, found by testing

CML promises that nack-guards are notified when `sync` is *interrupted*, not
only when they lose. Go has no way to kill a goroutine from outside, so the
only interruption a `Sync` can see is a panic. My first version had this bug:
if a guard maker panicked partway through the tree walk, nack-guards walked
earlier had already started their relay goroutines, and those goroutines waited
forever. A test that synced
`Choose(WithNack(...spawns a waiter...), Guard(func() { panic(...) }))` 100
times leaked 100 goroutines. The fix is a `defer` that closes every nack when
the commit never happened. I mention it because the article lists interruption
as a separate case, and in Go that case appears as `panic`.

## The part Go can't do: reclaiming goroutines

The article's strongest point is about garbage collection. In Racket, a thread
blocked on events that nothing else can reach is garbage, and the collector
frees it. Go 1.27 ships a `goroutineleak` profile in `runtime/pprof`. When you
request it, the runtime runs a special GC cycle, finds goroutines blocked on
unreachable concurrency primitives, and marks them with a new `_Gleaked` state.

I wanted to know whether it frees anything, so I leaked 200 goroutines. Each
one held 1 MiB and was blocked sending on a channel that no one else could
reach:

```
start:               goroutines=1    heap=0.1 MiB
after leak:          goroutines=201  heap=200.3 MiB
after leak profile:  leaked=200  goroutines=201  heap=200.3 MiB
again:               leaked=200  goroutines=201  heap=200.3 MiB
```

Every leaked goroutine was found correctly, and not one byte was freed. Go's
detector is a diagnostic. Racket's version is memory management.

This matters for CML style specifically. Look at `FetchEvt` above: the line
`go func() { <-nack; cancel() }()` runs for every sync. When the fetch *wins*,
the nack never fires, and that goroutine waits forever. I ran 1,000 syncs where
the guarded event won:

```
extra goroutines=1000, flagged leaked=1000
```

In Racket, that idiom is free, because the waiter becomes unreachable and the
collector removes it. In Go, the same idiom leaks one goroutine per *success*.
Your service gets slower as it handles more traffic successfully, which is
backwards. The leak detector will at least tell you about it.

The Go answer is to stop waiting on the nack and call a function. I added a
variant that takes an abort callback instead of a nack channel:

```go
func WithAbort[T any](mk func() (Event[T], func())) Event[T]
```

`Sync` runs the callback itself, after commit, for each losing subtree. Over
the same 1,000 syncs, it produced 0 extra goroutines and ran the callback
1,000 times, once per loser. The trade-off is that the callback runs inside
the caller's `Sync`, so it must not block. A loser that needs a blocking
reaction, like the print service's relay, still needs a goroutine and a nack.
The callback is the right default whenever the reaction is just "cancel the
context", which in Go is most of the time.

## So, is it unrealistic?

My conclusion is narrower than the article's in both directions:

- **The composability gap is a library problem, not a language one.** Choice,
  wrap, guard and nack fit in about 150 lines on top of `reflect.Select`. You pay
  4–5× per operation and lose compile-time `select` syntax. You gain events
  that can be returned, stored and combined, which is the article's main point.
- **The GC gap is a runtime problem, and it's real.** Go 1.27 can now *find*
  goroutines that are blocked on unreachable channels, but it doesn't free
  them. CML's most natural nack idiom depends on that freeing. Without it, you
  have to rewrite the idiom as callbacks.
- **I/O is still outside the system.** The article's `write-bytes-avail-evt`
  commits a write as the sync choice itself. In Go, I/O isn't something
  `select` can wait on. A goroutine relays the result over a channel, so the
  commit point is the relay and not the write. No library on top of `select`
  can change that.

I didn't benchmark Racket. These numbers compare Go events with Go `select`,
not Go with Racket. The full code, tests and examples are about 300 lines.
