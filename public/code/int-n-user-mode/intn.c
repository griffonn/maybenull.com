#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <ucontext.h>
#include <unistd.h>

static unsigned char *page;
static volatile int hits;
static int outfd;
struct rec { int sig, code, hits; long rip_off, trapno, err; long addr_off; unsigned long addr; };
static struct rec r;

static void h(int sig, siginfo_t *si, void *uc_) {
  ucontext_t *uc = uc_;
  long rip = uc->uc_mcontext.gregs[REG_RIP];
  hits++;
  if (hits == 1) {
    r.sig = sig; r.code = si->si_code;
    r.rip_off = rip - (long)page;
    r.trapno = uc->uc_mcontext.gregs[REG_TRAPNO];
    r.err = uc->uc_mcontext.gregs[REG_ERR];
    r.addr = (unsigned long)si->si_addr;
  }
  /* Return without touching RIP up to 3 times: a fault re-executes. */
  if (hits >= 3) { r.hits = hits; write(outfd, &r, sizeof r); _exit(0); }
}

int main(int argc, char **argv) {
  /* argv: hex bytes of the instruction under test */
  page = mmap(0, 4096, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  int n = 0;
  page[n++] = 0xB8; page[n++] = 0xFF; page[n++] = 0xFF; page[n++] = 0xFF; page[n++] = 0xFF; /* mov eax,-1 */
  int start = n;
  for (int i = 1; i < argc; i++) page[n++] = strtol(argv[i], 0, 16);
  int len = n - start;
  page[n++] = 0xC3;
  int fds[2]; pipe(fds);
  pid_t p = fork();
  if (p == 0) {
    outfd = fds[1];
    struct sigaction sa = {0}; sa.sa_sigaction = h; sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    int sigs[] = {SIGSEGV, SIGILL, SIGTRAP, SIGBUS, SIGFPE};
    for (int i = 0; i < 5; i++) sigaction(sigs[i], &sa, 0);
    ((void (*)(void))page)();
    r.hits = hits; r.sig = hits ? r.sig : 0; write(outfd, &r, sizeof r); _exit(0);
  }
  close(fds[1]);
  int st; waitpid(p, &st, 0);
  if (read(fds[0], &r, sizeof r) != sizeof r) { printf("died status=%x\n", st); return 0; }
  if (!r.sig) { printf("no signal (returned normally)\n"); return 0; }
  printf("sig=%-7s si_code=%d rip=%s trapno=%ld err=0x%lx si_addr=%s handler_calls=%s\n",
    sigabbrev_np(r.sig), r.code,
    r.rip_off == start ? "AT" : r.rip_off == start + len ? "AFTER" : "other",
    r.trapno, r.err,
    r.addr == 0 ? "0" : r.addr == (unsigned long)page + start ? "insn" : r.addr == (unsigned long)page + start + len ? "next" : "other",
    r.hits >= 3 ? "3+ (re-executes)" : "1 (continues)");
  return 0;
}
