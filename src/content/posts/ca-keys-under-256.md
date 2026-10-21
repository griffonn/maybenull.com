---
title: "Eight trusted CA keys whose private half was under 256"
description: "The eMudhra intermediates were 384-bit ECDSA certs with private scalars of 3 to 8 bits. I recovered all eight from the public certificates and scanned the rest of Mozilla's CA corpus to see if anything else was that weak."
date: 2026-10-21
draft: false
---

On 17 October someone signed the string "We have a problem here - Wayne" with
the private keys of eight eMudhra intermediate CA certificates — certificates
trusted, at the time, by Mozilla, Microsoft, Apple and Chrome. They didn't
steal the keys. They computed them from the certificates everyone already had.

The [disclosure](https://emudhra.we-have-a-problem.com/) says the keys are
"recoverable almost instantly with standard discrete-log techniques," and
[Mozilla's bug](https://bugzilla.mozilla.org/show_bug.cgi?id=2083449) adds the
crucial detail: these are P-384 keys whose private scalar was "an integer below
256." That's the kind of claim worth checking yourself, because the gap between
"technically recoverable" and "recoverable on a laptop in the time it takes to
read this sentence" is the whole story. So I pulled the public certificates and
tried.

## What a P-384 key is supposed to be

An ECDSA key on the P-384 curve is a private integer `d`, chosen uniformly at
random in `[1, n-1]` where `n` is the curve order — a 384-bit prime. The public
key is the point `Q = d·G`, `G` the standard generator. Security rests entirely
on the discrete-log problem: given `Q` and `G`, finding `d` should take on the
order of `√n ≈ 2^192` operations. That's the entire point of the curve. Nobody
has ever solved a discrete log at that size.

The catch is that the hardness is a property of `d` being *large and random*. The
curve doesn't enforce that. If you hand it a `d` of 5, it will cheerfully compute
`Q = 5·G` and the resulting public key looks exactly as random as any other —
the point's coordinates are two 384-bit numbers with no visible structure. You
cannot look at `Q` and tell that `d` is tiny. You have to go looking.

## Going looking

The textbook way to find a small `d` is baby-step/giant-step: precompute
`j·G` for `j` in `[1, m]` into a hash table keyed by the point's x-coordinate,
then walk `Q, Q-mG, Q-2mG, …` until one of them lands in the table. A hit at
giant step `i`, baby step `j`, recovers `d = im ± j` — the `±` because the
x-coordinate is the same for a point and its negation. With `m = √(bound)` it
finds any `d` below `bound` in roughly `2√bound` point additions.

For keys this small you don't even need that. If `d < 256`, you can multiply
`1·G, 2·G, …, 255·G` and compare — 255 multiplications, a few milliseconds. I
wrote the BSGS version anyway because I wanted to run it over *everything*, not
just the eight suspects, and BSGS is what you'd reach for if you didn't already
know the bound.

I fed it the eight eMudhra certificates straight from the incident's
`impacted-intermediates.json`. All eight resolved immediately. Here is the
private scalar's bit length for each — that is, `⌈log₂ d⌉`, so "3" means `d` is
one of `{4,5,6,7}`:

```
EM OV TLS CA - G3C   3 bits
EM EV TLS CA - G3C   6 bits
EM EV TLS CA - G3B   7 bits
EM OV CS  CA - G3B   7 bits
EM DV TLS CA - G3B   8 bits
EM DV TLS CA - G3C   8 bits
EM OV TLS CA - G3B   8 bits
EM OV CS  CA - G3C   8 bits
```

Every one is under 256, matching Mozilla's description exactly. The weakest, the
OV TLS G3C intermediate, has a private key you could write on a napkin. On a
curve whose security budget is 192 bits, these keys spend between 3 and 8. The
baby-step table never needed to grow past 256 entries; the whole set fell out in
about four seconds of single-threaded Python, most of which was parsing
certificates, not doing math.

A short digression on *why* this happened, from eMudhra's own account in the
Mozilla bug: a key-generation script was modified for a test wrap/unwrap on their
HSM, which generated a correct key pair and then a second, weaker one. Because
the key labels were similar, the weaker pair got used to build the CSRs by
mistake. So the "integer below 256" wasn't someone typing `d = 5`; it was a
malformed key that the HSM emitted and nobody caught, because — again — a bad
key is indistinguishable from a good one by looking at the certificate. The only
tell is doing the discrete log, and you only do that if you suspect something.

## Is anything else this broken?

The interesting question isn't whether these eight keys are weak — the
disclosure already told us that. It's whether this is a localized manufacturing
defect or the visible tip of something endemic. The nice thing about a BSGS scan
is that it's cheap enough to just run over the whole public CA ecosystem and
find out.

So I did. I pulled every intermediate and every root from Mozilla's CCADB
(the Common CA Database that backs the root programs), extracted the EC keys, and
ran the search over all of them at a generous bound of `2^24` — about 16 million,
far above the "256" these keys landed at, enough to catch anything merely
"suspiciously small." That corpus has **798 distinct EC public keys**: 407 on
P-256, 388 on P-384, and 3 on P-521. Across four processes the entire sweep took
about three seconds.

It flagged exactly eight keys. The same eight.

```
P-384   388 keys   bound 2^24   8 hits   (the eMudhra intermediates)
P-256   407 keys   bound 2^24   0 hits
P-521     3 keys   bound 2^24   0 hits
```

Nothing else in the publicly trusted CA set has a private key within a factor of
a million of being guessable this way. That's a reassuring result and a useful
one: it says this was a contained screwup at one issuer, not a common failure
mode of HSM key generation across the industry. (The scan only rules out
*small-scalar* weakness — it says nothing about shared RSA factors or biased
nonces, which are different attacks with different detection methods. It's one
lamp post, not the whole street.)

## The part that should make you a little uneasy

The thing I keep coming back to is that these certificates sat in Certificate
Transparency logs and CCADB from April 2025 until this month, fully trusted, and
the weakness was there the entire time in plain sight. Any one of the thousands
of people who mirror CCADB could have run the twenty lines of point arithmetic
above at any point and recovered the keys to eight CAs trusted by every major
browser. The disclosure notes there are no CT log entries for these intermediates
during that window, which means a CT-enforcing browser couldn't have been
attacked with a forged cert without the forgery itself being logged — but a
non-CT client had no such protection, and we have no way to know whether anyone
found this before Wayne did.

Verifying "the private key is recoverable" turned out to be the easy part. The
uncomfortable part is that the check which would have caught it — multiply a few
hundred points, see if the public key shows up — costs milliseconds, and could
have been a required step at issuance, and wasn't.

---

*Method: EC keys parsed from Mozilla CCADB's intermediate and root PEM exports
plus the incident's `impacted-intermediates.json`. Discrete logs by
baby-step/giant-step over NIST P-256/384/521 (affine arithmetic, scalar bound
`2^24`), run on a 4-core container. I did not recover or publish any private key
material; the certificates are revoked (`caCompromise`, 18 Oct 2026) and the
proof of compromise was already public. Sources:
[disclosure](https://emudhra.we-have-a-problem.com/),
[Mozilla bug 2083449](https://bugzilla.mozilla.org/show_bug.cgi?id=2083449).*
