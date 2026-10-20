#include <stdio.h>
#include <stdlib.h>
#include "swar.h"
static const char *AL="012569.";
static unsigned long tested, mism;
static void check(const uint8_t *s,long n){
  uint32_t a=0,b=0; int r1=swar_parse(s,n,&a), r2=ref_parse(s,n,&b); tested++;
  if(r1!=r2 || (r1&&a!=b)){ if(mism<40){ printf("MISMATCH len=%ld '",n); for(long i=0;i<n;i++) putchar(s[i]>=32&&s[i]<127?s[i]:'?'); printf("' swar=%d %08x ref=%d %08x\n",r1,a,r2,b);} mism++; }
}
int main(int argc,char**argv){
  int maxlen=atoi(argv[1]); uint8_t buf[64]; int K=7; int idx[32];
  for(int n=7;n<=maxlen;n++){ for(int i=0;i<n;i++) idx[i]=0;
    for(;;){ for(int i=0;i<n;i++) buf[i]=AL[idx[i]]; check(buf,n);
      int p=0; while(p<n && ++idx[p]==K){ idx[p]=0; p++; } if(p==n) break; }
    printf("len %d done, tested=%lu mism=%lu\n",n,tested,mism); fflush(stdout); }
}
