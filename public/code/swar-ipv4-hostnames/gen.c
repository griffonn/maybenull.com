/* emit mutated corpus (escaped) + expected ref result, for real-Swift differential */
#include <stdio.h>
#include <stdlib.h>
#include "swar.h"
static uint64_t rs=0x9E3779B97F4A7C15ull; static uint64_t rnd(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static FILE *fin,*fexp;
static void emit(const uint8_t*s,int n){ for(int i=0;i<n;i++){ if(s[i]>=33&&s[i]<127&&s[i]!='\\') fputc(s[i],fin); else fprintf(fin,"\\x%02x",s[i]); } fputc('\n',fin);
  uint32_t a; if(ref_parse(s,n,&a)) fprintf(fexp,"ok %u.%u.%u.%u\n",a>>24,(a>>16)&255,(a>>8)&255,a&255); else fprintf(fexp,"fail\n"); }
static int mkaddr(uint8_t *b){ int n=0; for(int k=0;k<4;k++){ int v=rnd()%256; int w=rnd()%4; char t[8]; int l;
   if(w==0) l=sprintf(t,"%03d",v); else l=sprintf(t,"%d",v); memcpy(b+n,t,l); n+=l; if(k<3) b[n++]='.'; } return n; }
int main(){ fin=fopen("diff_in.txt","w"); fexp=fopen("diff_exp.txt","w"); uint8_t b[64],c[64];
  static const uint8_t I[]={'0','1','2','5','6','9','.','/',':','-','+',' ',0,0xAE,0xB0,0xB9,0x1E,0x0E,0x3E,0xFF,0x80,'a','x'};
  for(int it=0;it<20000;it++){ int n=mkaddr(b); emit(b,n);
    for(int p=0;p<n;p++) for(int v=0;v<256;v+=1+(rnd()%7)){ memcpy(c,b,n); c[p]=v; if(v!='\n') emit(c,n);}
    for(int p=0;p<=n;p++){ memcpy(c,b,p); c[p]=I[rnd()%sizeof I]; memcpy(c+p+1,b+p,n-p); emit(c,n+1); }
    for(int k=0;k<20;k++){ memcpy(c,b,n); for(int j=0;j<2;j++) c[rnd()%n]=I[rnd()%sizeof I]; emit(c,n);} }
  for(int it=0;it<1000000;it++){ int n=rnd()%22; for(int i=0;i<n;i++) b[i]=I[rnd()%8]; emit(b,n);} }
