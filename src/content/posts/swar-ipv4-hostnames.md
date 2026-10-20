---
title: "A very fast IPv4 parser, fed hostnames instead"
description: swift-endpoint's new branchless SWAR IPv4 parser held up through 16 billion exhaustive inputs and 19 million fuzzed ones. It costs about 10 ns whether or not the input is an address, though, so on the hostnames a DNS library mostly sees it loses to a naive loop. Two early-exit checks halve that. Separately, a third of the strings it accepts mean a different address to a browser.
date: 2026-10-20
draft: false
---

Mahdi Bahrami's [A Very Fast IPv4 Parser, in Swift](https://blog.mahdibm.com/posts/a-very-fast-ipv4-parser-in-swift/)
is on the lobste.rs front page today. It walks through the IPv4 parser in
[swift-endpoint](https://github.com/swift-dns/swift-endpoint), the address and domain
library behind swift-dns. The parser is scalar SWAR: four overlapping 32-bit loads, an XOR
with `0x30` to turn digits into values and dots into `0x1E`, bit tricks to find the dots,
one carry-based check for invalid bytes and segments over 255, and a multiply by
`0x00640A01` to fold each segment's digits into a number. It has no lookup tables and no
loop. The post places it 2nd on an Apple M6 and 3rd on AMD Zen 5, close to simdzone's
SSE parser.

The post's benchmarks feed it only valid addresses, and the benchmark code isn't public
yet. That leaves three questions the post doesn't answer, and I could answer them on a
4-core Intel Xeon VM:

1. Is it correct on *invalid* input? Branchless validation is exactly where off-by-one-lane
   bugs hide.
2. How fast is it on what an endpoint library actually sees? Most strings handed to
   "is this an IP literal?" are hostnames.
3. What does it accept, and do other parsers read those strings the same way?

All code, data generators and raw output are [linked at the end](#reproduce-it). I tested
swift-endpoint at commit
[`c030b65`](https://github.com/swift-dns/swift-endpoint/tree/c030b6588540d577fbf86de19d4930008bd654b3)
with Swift 6.3.1.

## 1. Correctness: I couldn't break it

The parser is pure integer arithmetic, so I ported its body line for line to C
(`swar.h`), keeping Swift's masking-shift semantics (`&<<` takes the shift count mod the
bit width). Then I compared it against a 15-line reference: four runs of 1–3 ASCII digits,
each at most 255, joined by exactly three dots, leading zeros allowed (more on that below).

- **Exhaustive:** every string of length 7–12 over the alphabet `0 1 2 5 6 9 .`. That's
  16,148,031,144 inputs, chosen to hit the 255/256 boundary, empty segments, doubled dots
  and over-long segments. **0 mismatches.**
- **Mutation fuzz, under ASan/UBSan:** 200k random addresses (with and without zero
  padding), each byte replaced by all 256 values, single-byte inserts and deletes, 50
  random 1–3-byte corruptions using bytes chosen to fool the lane tricks (`0xAE` is a dot
  with the high bit set, `0x1E` is what a dot becomes after the XOR, `/` is a "dot" to the
  dot finder), plus 200M random strings of length 0–24. That's 917,896,888 inputs, every
  one in an exactly-sized heap buffer so ASan would catch a read past the end. **0
  mismatches, 0 out-of-bounds reads.**
- **The real library:** 19,347,314 of those inputs (470,205 of them valid) went through
  swift-endpoint's public `IPv4Address(textualRepresentation:)` in a Swift harness. Its
  output is **byte-identical** to the reference.

One thing I wanted to see: the function never checks `count <= 15`. Its middle bytes go
unread for longer inputs, so how does it reject `1.2.3.4` padded to 20 bytes? It turns out
the third-segment window is placed from the end of the string. For anything over 15 bytes,
the distance between the second dot and that window comes out negative, and the
`UInt(bitPattern:) <= 16` guard rejects it. The bound is real, just implicit. The random
0–24-byte strings above cover that path.

So the parser is correct. That matters for what follows: everything below is about cost
and semantics, not bugs.

## 2. Speed when the answer is "no"

"Branchless" here means it does the same work for every input of 7+ bytes. On valid
addresses that's the point. On `example.com` it does all the work too, and only then
returns false. A DNS library constantly asks whether a host string is an IP literal, and
usually the answer is no.

I built four corpora of 2²⁰ strings each. *shapes* is random valid addresses, as in the
original post. *real32* is 32 common addresses in shuffled order, also as in the original.
*domains* is swift-endpoint's own `top-domains.csv`: 200 names, median length 12, 18%
longer than 15 bytes. *mixed* is 90% domains and 10% addresses. Every number below is the
median of 5 runs, each the best of 7 passes, in ns per string. The loop itself costs about
0.7 ns.

| corpus  | swift-endpoint | + guards (Swift) | C port | C port + guards | naive loop | glibc `inet_pton` | Rust `Ipv4Addr` |
|---------|---------------:|-----------------:|-------:|----------------:|-----------:|------------------:|----------------:|
| shapes  | 9.9  | 10.1 | 8.2  | 8.5 | 31.4 | 43.5 | 41.8 |
| real32  | 10.0 | 10.1 | 8.2  | 8.4 | 29.3 | 38.8 | 40.0 |
| domains | **10.8** | **4.9** | 9.3 | **3.8** | **2.7** | 7.1 | 7.4 |
| mixed   | **13.4** | **6.2** | 12.6 | **5.4** | 7.3 | 12.2 | 12.3 |

*(Xeon @ 2.1 GHz VM, clang 18 `-O3 -march=native`, Swift 6.3.1 release. "Naive loop" is my
reference parser: a digit loop that returns at the first bad byte.)*

On valid input the SWAR parser is 4× faster than glibc and Rust std, which matches the
original post's ranking on a third microarchitecture. On hostnames, though, it's the
slowest thing in the table, at **4× the cost of a naive loop**,
because the naive loop rejects `google.com` at the `g`. On the mixed corpus it's slower
than glibc. The mixed corpus also costs more than either of its parts. That's branch
prediction: the caller's success/failure branch and the parser's three exit branches now
go both ways at random.

I checked the swift-endpoint numbers against the compiled code. The specialized
`IPv4Address.init?(textualRepresentation:)` is about 100 x86 instructions with three
conditional branches: `count < 7`, the segment-3 shift guard, and the combined validity
mask. The compiler moved the shift guard ahead of the validity check. There's nothing to
shave inside it.

The fix is outside it. Two checks the SWAR body already implies, but done first:

```swift
guard UInt(bitPattern: count &- 7) <= 8 else { return false }          // 7...15 bytes
guard unsafe span.bytes.unsafeLoad(fromUncheckedByteOffset: count &- 1,
                                   as: UInt8.self) &- 0x30 <= 9 else { return false } // ends in a digit
```

A valid address always ends in a digit. Almost no hostname does (TLDs are letters), so
the second check decides most non-addresses with one byte load and one well-predicted
branch. The length check alone doesn't help. It's another branch that goes both ways at
random (18% of the domains are too long) and it did slightly worse than no guard at all.
The last-byte check is the one that pays.

With both guards patched into the real library (`guard.patch`), the patched build is
**still byte-identical** to the reference on the 19.3M differential corpus. Hostnames go
from 10.8 to **4.9 ns** and the mixed corpus from 13.4 to **6.2 ns**. Valid addresses go
from 9.9 to 10.1 ns, which is within run-to-run noise. In C the same guards bring
hostnames to 3.8 ns, close to the naive loop's 2.7, without giving up the 4× win on
addresses.

The broader point is a benchmarking one. Benchmarking a validating parser only on valid
input measures half its job. That's even more true for a branchless one, because it gives
up early exit on purpose.

## 3. What "valid" means: leading zeros

swift-endpoint accepts leading zeros and reads them as decimal, so `010.020.030.040` is
10.20.30.40. This is deliberate: a comment in its tests cites RFC 6943 and notes that
Apple's `inet_pton` behaves the same way while glibc rejects. Here is how the parsers on
this box handle a few strings:

| input        | swift-endpoint | glibc `inet_pton` | Go `netip`, Rust std, Python `ipaddress` | glibc `inet_aton` | WHATWG URL (Node 22) |
|--------------|-----------|---------|---------|-----------|-----------|
| `192.0.2.01` | 192.0.2.1 | reject  | reject  | 192.0.2.1 | 192.0.2.1 |
| `010.0.0.1`  | **10.0.0.1** | reject | reject | **8.0.0.1** | **8.0.0.1** |
| `012.0.0.1`  | **12.0.0.1** | reject | reject | **10.0.0.1** | **10.0.0.1** |
| `08.0.0.1`   | 8.0.0.1   | reject  | reject  | reject    | reject    |
| `0177.0.0.1` | reject    | reject  | reject  | 127.0.0.1 | 127.0.0.1 |
| `127.1`      | reject    | reject  | reject  | 127.0.0.1 | 127.0.0.1 |

The strict parsers reject every zero-padded segment. The lenient family (`inet_aton`, and
the WHATWG URL parser that browsers and Node's `URL` follow)
reads a leading zero as **octal**. swift-endpoint is the only one here that accepts the
string *and* reads it as decimal.

I counted how much of its input space this affects. Each segment can be written
366 ways: the 256 canonical values, plus zero-padded forms like `07`, `007` and `042`.
I ran all 366 through Node's `URL`. 272 give the same value (the canonical ones, and
padded values below 8, where octal and decimal agree). 38 make WHATWG reject the host
(a padded segment containing 8 or 9). **56 give a different number**: `010`–`077`, read
as 8–63. Across four segments:

```text
strings swift-endpoint accepts      366^4 = 17,944,209,936
  of which canonical                256^4 =  4,294,967,296   (23.9%)
  WHATWG reads the same address     272^4                     30.5%
  WHATWG reads a DIFFERENT address  328^4 - 272^4             34.0%
  WHATWG rejects                    366^4 - 328^4             35.5%
```

So a third of the strings this parser accepts name a different machine to a browser.
Mostly that's harmless. It becomes a bug in one pattern: a server that validates a
user-supplied URL host with one parser ("is this a private address?") and then fetches it
with another. That's the parser-differential SSRF behind CVE-2021-29923 in Go's `net` and
CVE-2021-29921 in Python's `ipaddress`. Both languages answered by rejecting leading zeros
outright.

Working it through for this parser, the octal readings that fit in three characters top
out at `077` = 63. So the only private range an attacker reaches this way is **10/8**,
via `012`. `012.0.0.5` checks out as 12.0.0.5, a public AT&T address, and the WHATWG
fetcher connects to 10.0.0.5. Loopback (`0177`), 172.16/12 (`0254`), 192.168/16 (`0300`)
and the metadata address 169.254.169.254 (`0251.0376…`) all need four-character segments,
and swift-endpoint rejects those. That limits the damage, but 10/8 is the range most
internal networks live in. If you're validating hosts in a Swift server, either
canonicalize (parse, then re-serialize and compare to the input) or reject any segment
that starts with `0` and has more than one digit. Go and Python made the same change.

## What I'd take away

- The SWAR parser is correct. Sixteen billion exhaustive strings, 900M sanitized fuzz
  cases and a 19M-case run against the real library found nothing. The implicit 15-byte
  bound holds.
- It costs ~10 ns on this machine for any input of 7+ bytes. On hostname-heavy traffic
  that's 4× a naive loop, and slower than glibc on a 90/10 mix.
- A last-byte-is-a-digit check up front (plus a length check) halves that cost and leaves
  valid-address speed and the output unchanged. It's a two-line patch.
- Accepting `010` as decimal is a defensible choice for a DNS library. In a URL-validation
  path it disagrees with browsers on 34% of the strings it accepts, and that's enough for
  an SSRF into 10/8.

## Caveats

- One x86 machine, a shared cloud VM at 2.1 GHz. Absolute numbers are slower than the
  original post's M6 and Zen 5 figures, and individual runs moved by up to ±15%. The
  ordering of columns never changed.
- The C port is my translation. The Swift harness numbers and the differential results
  use the real library, built with its default traits switched off. The harness wraps the
  public init in a `@inline(never)` function. That costs slightly more than inlining
  would, but it applies equally to both the patched and unpatched builds.
- My corpora aren't the post's (its benchmark isn't published), and the domain list is
  only 200 names. Production traffic will have a different hostname/address ratio. The
  break-even depends on that ratio.
- I didn't test other WHATWG edge cases, such as trailing dots, which WHATWG accepts and
  swift-endpoint rejects. Rejecting can't produce a wrong address, only a hostname lookup.

## Reproduce it

Everything is in [`/code/swar-ipv4-hostnames/`](https://github.com/griffonn/maybenull.com/tree/main/public/code/swar-ipv4-hostnames):

- `swar.h` holds the C port and the reference parser. `exh.c` runs the exhaustive test
  (`./exh 12`; output in `exhaustive.txt`). `fuzz.c` is the ASan mutation fuzzer.
- `gen.c` writes the 19.3M-line differential corpus and the expected results. Run
  `harness/` (a SwiftPM package expecting swift-endpoint at `../src/swift-endpoint`) with
  `H check diff_in.txt | cmp - diff_exp.txt`.
- `corpora.py` builds the benchmark sets from swift-endpoint's `top-domains.csv`.
  `bench.c` and `rbench.rs` give the C, glibc and Rust columns, `H bench <file> 3` gives
  the Swift ones, and `guard.patch` is the two-line change.
- `cmp.c`, `gocmp.go` and `cases.txt` produce the semantics table.

*Sources: [Mahdi Bahrami, "A Very Fast IPv4 Parser, in Swift"](https://blog.mahdibm.com/posts/a-very-fast-ipv4-parser-in-swift/),
[swift-endpoint `c030b65`](https://github.com/swift-dns/swift-endpoint/blob/c030b6588540d577fbf86de19d4930008bd654b3/Sources/IPAddress/IPv4Address/String+IPv4Address.swift),
[WHATWG URL Standard, IPv4 parser](https://url.spec.whatwg.org/#concept-ipv4-parser),
[RFC 6943 §3.1.1](https://www.rfc-editor.org/rfc/rfc6943#section-3.1.1),
[CVE-2021-29923](https://nvd.nist.gov/vuln/detail/CVE-2021-29923),
[CVE-2021-29921](https://nvd.nist.gov/vuln/detail/CVE-2021-29921).
The tests, benchmarks, counts and conclusions are my own.*
