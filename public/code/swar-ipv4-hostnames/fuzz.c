#include <stdio.h>
#include <stdlib.h>
#include "swar.h"
static unsigned long tested, mism;
static uint64_t rs=88172645463325252ull; static uint64_t rnd(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static void check(const uint8_t *src,long n){
  uint8_t *s=malloc(n?n:1); memcpy(s,src,n);   /* exact-size buffer so ASan catches OOB reads */
  uint32_t a=0,b=0; int r1=swar_parse(s,n,&a), r2=ref_parse(s,n,&b); tested++;
  if(r1!=r2 || (r1&&a!=b)){ if(mism<40){ printf("MISMATCH len=%ld '",n); for(long i=0;i<n;i++) printf(s[i]>=32&&s[i]<127?"%c":"\\x%02x",s[i]); printf("' swar=%d %08x ref=%d %08x\n",r1,a,r2,b);} mism++; }
  free(s);
}
static int mkaddr(uint8_t *b){ int n=0; for(int k=0;k<4;k++){ int v=rnd()%256; int w=rnd()%4; char t[8]; int l;
   if(w==0) l=sprintf(t,"%03d",v); else l=sprintf(t,"%d",v); memcpy(b+n,t,l); n+=l; if(k<3) b[n++]='.'; } return n; }
int main(){
  uint8_t b[64], c[64]; static const uint8_t I[]={'0','1','2','5','6','9','.','/',':','-','+',' ',0,0xAE,0xB0,0xB9,0x1E,0x0E,0x3E,0x20,0xFF,0x80,0x7F,'a','x'};
  for(int it=0;it<200000;it++){ int n=mkaddr(b); check(b,n);
    for(int p=0;p<n;p++) for(int v=0;v<256;v++){ memcpy(c,b,n); c[p]=v; check(c,n); }
    for(int p=0;p<=n;p++){ /* insert / delete */ memcpy(c,b,p); c[p]=I[rnd()%sizeof I]; memcpy(c+p+1,b+p,n-p); check(c,n+1);
      if(p<n){ memcpy(c,b,p); memcpy(c+p,b+p+1,n-p-1); check(c,n-1);} }
    for(int k=0;k<50;k++){ memcpy(c,b,n); int m=1+rnd()%3; for(int j=0;j<m;j++) c[rnd()%n]=I[rnd()%sizeof I]; check(c,n); }
  }
  for(long it=0;it<200000000;it++){ int n=rnd()%25; for(int i=0;i<n;i++) b[i]=I[rnd()%9]; check(b,n); }
  printf("tested=%lu mism=%lu\n",tested,mism);
}
