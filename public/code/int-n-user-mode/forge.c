#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>
static void h(int sig, siginfo_t *si, void *uc_) {
  ucontext_t *uc = uc_;
  printf("forged: sig=%s si_code=%d si_addr=%p trapno=%lld err=0x%llx\n", sigabbrev_np(sig), si->si_code, si->si_addr,
    uc->uc_mcontext.gregs[REG_TRAPNO], uc->uc_mcontext.gregs[REG_ERR]);
}
static long n;
static void skip(int sig, siginfo_t *si, void *uc_) { ucontext_t *uc = uc_; n++; uc->uc_mcontext.gregs[REG_RIP] += 2; }
int main(void) {
  struct sigaction sa = {0}; sa.sa_sigaction = h; sa.sa_flags = SA_SIGINFO;
  sigaction(SIGILL, &sa, 0);
  siginfo_t si; memset(&si, 0, sizeof si);
  si.si_signo = SIGILL; si.si_code = ILL_ILLOPN; si.si_addr = (void*)0x401000;
  long rc = syscall(SYS_rt_sigqueueinfo, getpid(), SIGILL, &si);
  printf("rt_sigqueueinfo to self rc=%ld\n", rc);
  /* timing: ud2 emulated by advancing RIP */
  sa.sa_sigaction = skip; sigaction(SIGILL, &sa, 0);
  struct timespec a, b; int N = 200000;
  clock_gettime(CLOCK_MONOTONIC, &a);
  for (int i = 0; i < N; i++) __asm__ volatile("ud2");
  clock_gettime(CLOCK_MONOTONIC, &b);
  double ns = ((b.tv_sec-a.tv_sec)*1e9 + (b.tv_nsec-a.tv_nsec)) / N;
  printf("ud2 -> SIGILL -> skip round trip: %.0f ns (%ld signals)\n", ns, n);
  clock_gettime(CLOCK_MONOTONIC, &a);
  for (int i = 0; i < N; i++) syscall(SYS_getppid);
  clock_gettime(CLOCK_MONOTONIC, &b);
  printf("getppid syscall: %.0f ns\n", ((b.tv_sec-a.tv_sec)*1e9 + (b.tv_nsec-a.tv_nsec)) / N);
}
