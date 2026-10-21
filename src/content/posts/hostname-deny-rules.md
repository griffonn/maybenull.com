---
title: "A hostname deny rule only binds a client who agrees to it"
description: "Anthropic found hostname deny rules failing open when reverse DNS is off. I traced the same rule through OpenSSH, PostgreSQL and Apache with DNS on, and the only client it ever blocks is one who publishes a matching, forward-confirmed PTR record."
date: 2026-10-21
draft: false
---

Anthropic's Frontier Red Team published a write-up this week on
[using Claude to shrink the attack surface of code](https://www.anthropic.com/research/attack-surface-reduction).
One finding stuck with me. In four unrelated projects, access rules written
with hostnames silently failed open, because reverse DNS lookups were off
(the usual default). The server compared a pattern like `*.evil.example`
against a bare IP address, so the deny rule could never fire.

The post doesn't name the four projects, and the obvious fix is "turn DNS on".
I wanted to check whether that fix holds up. So I read the host-matching code
in three servers that support hostname rules, OpenSSH, PostgreSQL and Apache
httpd, and modeled what happens when DNS is **on**. Short answer: switching DNS
on fixes the rule for clients who play along, and only for them. A hostname
deny rule fires only when the client publishes a PTR record that matches it
*and* a forward record that confirms it. Anyone who would rather not be
blocked doesn't publish them.

## Why DNS-based rules are asymmetric

To go from an IP address to a hostname, a server does a reverse (PTR) lookup.
Whoever controls the reverse zone for an address can make that lookup return
any name. Usually that's the netblock owner, and many VPS hosts hand PTR
control to their customers. Everyone has known this for decades, which is
why careful servers use *forward-confirmed reverse DNS*. They look up the PTR
name forward again and accept it only if the result includes the original IP.
An attacker can set a PTR saying `ws1.corp.example`, but they can't make
`corp.example`'s own nameservers agree.

That defence is built for **allow** rules. It stops a stranger from posing as
a trusted name. Look at what each server does when the check fails, though.
None of them reject the connection. They fall back to "this client has no
hostname" and go on to the next rule. For an allow rule, that fallback is
fail-closed. For a deny rule it's fail-open. The forward-confirmation step
that guards allow rules becomes a way out for deny rules: an attacker who
*is* in `evil.example` only has to break their own forward record.

## What the code actually does

**OpenSSH** (`openssh-portable`, head of 18 Oct 2026). `allowed_user()` in
`auth.c` gets the client's name from `auth_get_canonical_hostname()`. With
`UseDNS no` (the default), that function just returns the IP string. With
`UseDNS yes` it calls `ssh_remote_hostname()` in `packet.c`, which hands back
the IP in four situations: no PTR, a PTR that is itself a numeric address
(the "Nasty PTR record" log line), a forward lookup that fails, or a forward
lookup that doesn't include the client's address. After that,
`match_host_and_ip()` in `match.c` runs the pattern against both strings. A
hostname glob that isn't an address goes through `addr_match_list()`, fails
CIDR parsing, gets glob-matched against `198.51.100.23`, misses, and the
rule is skipped.

**PostgreSQL** (`hba.c`, `check_hostname()`). Hostname entries in
`pg_hba.conf` always trigger a lookup, and no setting turns that off. A failed
reverse lookup, a name that doesn't match, or a forward lookup that doesn't
confirm all return `false`. That means *this line doesn't apply*, and
PostgreSQL moves on to the next line. A `reject` line keyed on `.evil.example`
followed by `0.0.0.0/0 scram-sha-256` therefore sends every unconfirmed client
straight to password auth. The only trace is a `DEBUG2` message.

**Apache httpd** (`mod_authz_host.c`, `mod_authz_core.c`). `Require host`
always uses `REMOTE_DOUBLE_REV`, i.e. forward-confirmed reverse DNS, whatever
`HostnameLookups` says. If it can't get a confirmed name, it logs an error and
returns `AUTHZ_DENIED`. The usual way to write a deny list in httpd 2.4 is
`Require not host evil.example` inside `<RequireAll>`. In `mod_authz_core.c`,
negating `DENIED` gives `NEUTRAL`, not `GRANTED`, and the source comment
explains the reasoning: a failed check shouldn't be enough to *grant* access.
That's correct for a grant. Here it means "couldn't resolve" becomes
"no objection", and `Require all granted` lets the client in.

## The matrix

I ported the decision logic of each server (the matching and fallback rules,
not the network code) to about 190 lines of Python. Then I ran the same
attacker IP through six DNS setups. The deny rule targets `*.evil.example`,
and the allow rule admits only `*.corp.example`.

```
Deny rule against *.evil.example   (BLOCK = rule fired)
scenario                                        ssh no  ssh yes     pg  httpd
S1 honest: PTR=bot.evil.example, fwd confirms     pass    BLOCK  BLOCK  BLOCK
S2 no PTR record                                  pass     pass   pass   pass
S3 PTR=bot.evil.example, fwd broken               pass     pass   pass   pass
S4 PTR=bot.evil.example, fwd -> other IP          pass     pass   pass   pass
S5 PTR=cdn.innocent.example, fwd confirms         pass     pass   pass   pass

Allow rule for *.corp.example   (ADMIT = attacker got in)
scenario                                        ssh no  ssh yes     pg  httpd
S1 .. S5 (as above)                                 no       no     no     no
S6 PTR claims ws1.corp.example, fwd no              no       no     no     no
```

Out of 20 deny cells, one blocks: the attacker who publishes a matching PTR
record and keeps its forward record consistent. The allow rule never lets an
outsider in, under any DNS setup. That's the asymmetry. Each server is doing
exactly what it was built to do. The built-in assumption is that DNS evidence
can only ever *add* trust. A deny rule needs DNS evidence that takes trust
away, and the client gets to decide whether that evidence exists.

S5 is the realistic one, not S3. Say you deny `*.cheap-vps.example` because
abuse keeps coming from that hosting provider. Someone rents a box there,
points its PTR at a domain they own, adds a matching A record, and your rule
no longer sees them. No misconfiguration needed, just two DNS records.

## One more silent no-op in OpenSSH

While tracing `match.c` I noticed a second pattern that can never fire, and
this one fails regardless of DNS. `match_pattern_list()` returns `-1` for a
negated hit and `1` only when there was a *positive* hit. So a rule made only
of negations, like

```
DenyUsers *@!*.corp.example     # intent: deny anyone not from corp
```

never denies anyone. `allowed_user()` denies when `match_user()` returns
non-zero, and `-1` is non-zero, so I traced it twice. The `-1` never reaches
it: `match_host_and_ip()` turns a negative hostname hit into `0` ("no match"),
and a client from outside corp gets no positive hit at all, which is also `0`.
In my model every scenario returns `0`, with DNS on or off, and so does a
properly confirmed `ws1.corp.example`. The `ssh_config(5)` PATTERNS section
does say a negation alone never produces a positive match. But sshd's
config-time check (`match_user()` called with NULL arguments) accepts the
pattern without a warning, the same way it accepts `mallory@*.evil.example`
under `UseDNS no`. What this rule should say is `AllowUsers *@*.corp.example`.
That's an allow rule, so it fails closed.

## What to do instead

- **Write deny rules against addresses, never names.** CIDR blocks are facts
  about the packet. A PTR record is a claim made by whoever controls the
  netblock's reverse zone.
- **If you have to filter by name, turn it around.** Allow the names you
  trust and default-deny the rest. Every implementation above gets that
  direction right.
- **Audit for the shape, not the setting.** Look for any negated or deny-type
  rule whose operand is a hostname or domain: `DenyUsers`/`DenyGroups` with
  `@host`, `Match Host` blocks that *tighten* settings, `pg_hba.conf`
  `reject` lines with hostnames, `Require not host`. Each one is advisory.

## What's mine and what isn't

The four fail-open projects and the `UseDNS`-off framing come from
[Anthropic's post](https://www.anthropic.com/research/attack-surface-reduction).
The DNS-on analysis, the cross-server comparison, the matrix and the
negation-only finding are mine. They come from reading
[openssh-portable](https://github.com/openssh/openssh-portable)
(`auth.c`, `match.c`, `addrmatch.c`, `packet.c`),
[PostgreSQL's `hba.c`](https://github.com/postgres/postgres/blob/master/src/backend/libpq/hba.c),
and httpd's
[`mod_authz_host.c`](https://github.com/apache/httpd/blob/trunk/modules/aaa/mod_authz_host.c)
and
[`mod_authz_core.c`](https://github.com/apache/httpd/blob/trunk/modules/aaa/mod_authz_core.c).
I didn't run the daemons themselves. The table comes from a port of their
decision logic, and a few details are simplified: Apache's `in_domain()`
handles more edge cases than my version, and I used `fnmatch` in place of
OpenSSH's glob engine for `*` and `?`. The conclusion depends only on the
fallback behaviour, which is the same in all three: when a hostname can't be
confirmed, the rule is skipped and the server never rejects the connection
for that reason. Each fallback is a few lines of code that you can check
against the files linked above.
