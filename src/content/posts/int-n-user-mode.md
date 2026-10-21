---
title: "All 256 int n instructions, run from Linux user mode"
description: Raymond Chen explains why int 6 can't stand in for ud2. I ran every int n from a 64-bit Linux process to see what actually comes back. 253 of them turn into the same SIGSEGV, with the vector number tucked into an error code. Three get through. And a process can forge a SIGILL that fools si_code but not trapno.
date: 2026-10-21
draft: false
---

Raymond Chen's post on The Old New Thing this week,
[Why do we need ud2 when int 6 is right there?](https://devblogs.microsoft.com/oldnewthing/20261019-00/?p=112781/),
gives two reasons a program can't fake an invalid-opcode fault with `int 6`. First, a
software interrupt is a *trap*: the saved instruction pointer points past it. A real #UD
is a *fault*, so the saved pointer points at the bad instruction. A handler that decodes
"the instruction that failed" would decode the wrong bytes. Second, the operating system's
interrupt table doesn't let user mode run `int 6` in the first place. The post stops
there, and doesn't say what you get instead.

That's easy to check, so I checked it, for all 256 vectors and not just 6. This was a
64-bit process on Linux 6.18 (a Firecracker VM on an Intel Xeon). Here's what each one
does, what a signal handler can and can't tell apart, and one way to forge a SIGILL that
the kernel lets through.

## The probe

For each test, `intn.c` writes `mov eax, -1; <instruction bytes>; ret` into an
executable page. A forked child calls it with handlers installed for SIGSEGV, SIGILL,
SIGTRAP, SIGBUS and SIGFPE. The handler records:

- the signal and `si_code`
- `si_addr`
- the saved RIP from the `ucontext`, compared with the instruction's start and end
- `REG_TRAPNO` and `REG_ERR`, the CPU's exception vector and error code, which Linux
  copies into the signal frame

Then it **returns without touching RIP** and counts how often it runs again. If the
handler fires again, the CPU went back and re-ran the instruction, so it was a fault. If
it fires once and execution carries on, it was a trap. That's Chen's fault/trap
distinction, seen from user space. (The `mov eax, -1` is there so that `int 0x80` makes
an invalid system call and does nothing.)

## The headline cases

```
bytes   instruction    signal  si_code  saved RIP  trapno  err    si_addr  handler runs
0F 0B   ud2            SIGILL  2        AT         6       0      insn     repeats (fault)
CD 06   int 6          SIGSEGV 128      AT         13      0x32   0        repeats (fault)
CC      int3           SIGTRAP 128      AFTER      3       0      0        once (trap)
CD 03   int 3          SIGTRAP 128      AFTER      3       0      0        once (trap)
F1      int1 / icebp   SIGTRAP 1        AFTER      1       0      next     once (trap)
CD 01   int 1          SIGSEGV 128      AT         13      0xa    0        repeats (fault)
CD 04   int 4          SIGSEGV 128      AFTER      4       0      0        once (trap)
CE      into           SIGILL  2        AT         6       0      insn     repeats (fault)
CD 80   int 0x80       (none: system call, returned -ENOSYS)
```

`si_code` 128 is `SI_KERNEL`, and 2 is `ILL_ILLOPN`.

The first surprise is that **`int 6` never reaches the invalid-opcode handler**, not even
the wrong way round. It becomes a general-protection fault (vector 13) *on the `int`
instruction itself*. So the "RIP points past it" problem from the post never comes up on
Linux. The CPU checks privilege before it delivers the interrupt. The privilege check
fails, and that failure is an ordinary fault, so RIP stays on the `int`. A handler that
does nothing gets called forever.

The error code says why. On a #GP caused by an interrupt gate, bits 3 and up hold the
vector, bit 1 says "this index is in the IDT", and bit 0 says the event was external:

```
err = 0x32 = 0b0011_0010
              └─┬──┘ │└ EXT = 0
                │    └─ IDT = 1
                └─ index = 6
```

Across all 253 vectors that end in #GP, the error code was exactly `(n << 3) | 2`, with
no exceptions (checked by script against `all.txt`). Linux turns all of them into
SIGSEGV with `si_addr = 0`, which is also what a #GP from a non-canonical address looks
like. So **a SIGSEGV handler can't tell `int 6` from a wild-pointer crash using
`siginfo` alone**. It needs `uc_mcontext.gregs[REG_ERR]`, and from that it can recover
which `int` you ran.

## The three that get through

The full sweep over `CD 00`–`CD FF` gives three outcomes:

```
253  SIGSEGV  #GP (trapno 13), RIP AT the int, re-executes
  1  SIGTRAP  vector 3,  RIP AFTER, continues     (int 3)
  1  SIGSEGV  vector 4,  RIP AFTER, continues     (int 4)
  1  no signal                                     (int 0x80)
```

These are exactly the gates that Linux's
[`arch/x86/kernel/idt.c`](https://github.com/torvalds/linux/blob/master/arch/x86/kernel/idt.c)
creates with its `SYSG` macro, which sets descriptor privilege level 3:
`X86_TRAP_BP`, `X86_TRAP_OF` and `IA32_SYSCALL_VECTOR`. Every other gate is DPL 0. Each
of the three has a small quirk.

**`int 4` is the only way into the overflow handler in 64-bit mode.** The instruction it
was meant for, `into` (`CE`), isn't valid in long mode. It raises #UD and arrives as an
ordinary SIGILL (row 8 above). But the gate is still open to user mode, so an explicit
`CD 04` reaches `exc_overflow`. In
[`traps.c`](https://github.com/torvalds/linux/blob/master/arch/x86/kernel/traps.c) that
handler sends SIGSEGV. The result is a SIGSEGV that behaves like a trap: RIP is past the
instruction, and a handler that does nothing just continues. Nothing else in the sweep
does that. Code that assumes "SIGSEGV means a fault, and returning means a retry" is
wrong here, though only for a program that goes out of its way to run `int 4`.

**`CC` and `CD 03` look identical to the handler.** Both report trapno 3 and RIP after
the instruction. Nothing in the frame says whether the breakpoint was one byte or two.
Any tool that finds the breakpoint address by subtracting one from the reported PC (the
usual x86 convention, since debuggers insert the one-byte `CC`) lands on the `03` byte,
in the middle of the instruction, when the program used the two-byte form.

**`int 1` and `int1` aren't the same thing.** `CD 01` goes through a DPL-0 gate and
becomes #GP. `F1` (ICEBP) is its own opcode, and the CPU treats it as a debug exception
that skips the gate privilege check, so it comes back as SIGTRAP with `si_code`
`TRAP_BRKPT`. It's also the only row whose `si_addr` points *after* the instruction.

## So can user space fake a #UD at all?

Chen's conclusion is that only a real undefined instruction produces a real #UD. At the
CPU level that holds. On Linux, though, a user-space #UD handler is a SIGILL handler, and
SIGILL is just a signal, so the question becomes whether a process can forge one.

`kill` and `raise` are out: they set `si_code` to `SI_USER` or `SI_TKILL`, which
plainly isn't `ILL_ILLOPN`. But `rt_sigqueueinfo` lets the caller supply the whole
`siginfo_t`. The kernel only refuses a non-negative `si_code` when the target is
*another* process. A process signalling itself can set anything. `forge.c` does exactly
that:

```c
si.si_signo = SIGILL; si.si_code = ILL_ILLOPN; si.si_addr = (void*)0x401000;
syscall(SYS_rt_sigqueueinfo, getpid(), SIGILL, &si);   // returns 0
```

The handler gets `sig=SIGILL si_code=2 si_addr=0x401000`, which is field for field what a
real `ud2` at that address would produce. The forgery shows only in the machine context:
**`REG_TRAPNO` is 0**, not 6, and RIP is wherever the system call returned, not at
`si_addr`. So the Linux version of Chen's rule is: `siginfo` can be forged, and the
trap number can't. A SIGILL-based instruction emulator that trusts `si_addr`, rather
than checking trapno and taking the address from the saved RIP, can be steered into
decoding whatever bytes the forger points it at. In practice that's the process attacking
itself, so it matters most when the "process" is a sandbox or a runtime hosting code it
doesn't trust.

## What emulating through #UD costs

The one real use Chen mentions for decoding the faulting instruction is emulating
instructions the CPU lacks. User-space programs on Linux still do this through SIGILL.
I timed a loop of `ud2` with a handler that adds 2 to RIP:

```
ud2 -> SIGILL -> handler -> sigreturn   ~8.0–8.3 µs   (200k iterations, 3 runs)
getppid() system call                   ~0.1 µs
```

That's about 80 system calls' worth of time for each emulated instruction. Some of it
comes from the VM: every trip through the kernel goes past the hypervisor, and saving and
restoring the extended register state in the signal frame isn't free. Bare-metal numbers
will be lower, but the ratio is why emulation through traps is a compatibility fallback
and not something you'd put in a hot loop.

## What I measured and what I assumed

Everything in the tables was measured on this one kernel and CPU (Linux 6.18, Intel
Xeon, under Firecracker). The gate list matches the kernel source today, but `int 0x80`
can be turned off with the `ia32_emulation=` boot parameter. On such a kernel I'd expect
vector 0x80 to join the #GP group, but I didn't test that. The error-code layout and the
fault/trap behaviour come from the CPU. Windows and other kernels choose their own
gates, and Chen notes that the ones he covers deny `int 6` to user mode too. I didn't
check how they report it. I didn't run gdb against `CD 03`, so the off-by-one is a
consequence of the frame being identical, not something I observed in a debugger.

## Reproduce it

The code is in
[`/code/int-n-user-mode/`](https://github.com/griffonn/maybenull.com/tree/main/public/code/int-n-user-mode):

- `intn.c` is the probe. Build it with `gcc -O1 -o intn intn.c`, then run
  `./intn cd 06`, `./intn 0f 0b` and so on. Sweep with
  `for i in $(seq 0 255); do printf '%02x ' $i; ./intn cd $(printf %02x $i); done`.
- `all.txt` is the raw output of that sweep.
- `forge.c` holds the `rt_sigqueueinfo` forgery and the timing loops.
