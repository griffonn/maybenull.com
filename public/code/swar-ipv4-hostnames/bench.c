#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arpa/inet.h>
#include "swar.h"
static uint8_t *buf; static long *off,*len; static long n;
static void load(const char*f){ FILE*fp=fopen(f,"rb"); fseek(fp,0,SEEK_END); long sz=ftell(fp); rewind(fp); buf=malloc(sz+16); fread(buf,1,sz,fp); fclose(fp);
  n=0; for(long i=0;i<sz;i++) if(buf[i]=='\n') n++; off=malloc(n*8); len=malloc(n*8); long s=0,k=0;
  for(long i=0;i<sz;i++) if(buf[i]=='\n'){ off[k]=s; len[k]=i-s; buf[i]=0; k++; s=i+1; } }
__attribute__((noinline)) static int f_swar(const uint8_t*s,long c,uint32_t*o){ return swar_parse(s,c,o); }
__attribute__((noinline)) static int f_swar_guard(const uint8_t*s,long c,uint32_t*o){ if((unsigned long)(c-7)>8) return 0; return swar_parse(s,c,o); }
__attribute__((noinline)) static int f_swar_guard2(const uint8_t*s,long c,uint32_t*o){ if((unsigned long)(c-7)>8 || (uint8_t)(s[c-1]-'0')>9) return 0; return swar_parse(s,c,o); }
__attribute__((noinline)) static int f_scalar(const uint8_t*s,long c,uint32_t*o){ return ref_parse(s,c,o); }
__attribute__((noinline)) static int f_pton(const uint8_t*s,long c,uint32_t*o){ return inet_pton(AF_INET,(const char*)s,o)==1; }
typedef int(*F)(const uint8_t*,long,uint32_t*);
static double run(F f){ double best=1e30; uint32_t sink=0;
  for(int r=0;r<7;r++){ struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    for(long i=0;i<n;i++){ uint32_t x=0; sink+= f(buf+off[i],len[i],&x)? x:1; }
    clock_gettime(CLOCK_MONOTONIC,&b); double ns=((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/n; if(ns<best) best=ns; }
  if(sink==42) puts(""); return best; }
int main(int argc,char**argv){ load(argv[1]);
  printf("%-10s swar=%.2f swar+len=%.2f swar+len+lastdigit=%.2f scalar=%.2f glibc_pton=%.2f\n", argv[1]+2,
    run(f_swar),run(f_swar_guard),run(f_swar_guard2),run(f_scalar),run(f_pton)); }
