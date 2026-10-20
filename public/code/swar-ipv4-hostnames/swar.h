#include <stdint.h>
#include <string.h>
static inline uint32_t ld32(const uint8_t *p, long off){ uint32_t v; memcpy(&v,p+off,4); return v; }
static inline int ctz32(uint32_t x){ return x? __builtin_ctz(x):32; }
static inline int clz32(uint32_t x){ return x? __builtin_clz(x):32; }
/* faithful port of swift-endpoint parseIPv4 @ c030b65; returns 1 on success, addr in network order bytes a.b.c.d -> (a<<24|...) */
static inline int swar_parse(const uint8_t *s, long count, uint32_t *out){
  if (count < 7) return 0;
  const uint64_t m30=0x3030303030303030ull; const uint32_t m30u=0x30303030u;
  uint32_t w1=ld32(s,0), window1=w1^m30u;
  uint32_t w2=ld32(s,(count>>1)&4);
  uint64_t window12=((uint64_t)w1|((uint64_t)w2<<32))^m30;
  uint32_t w4=ld32(s,count-4), window4=w4^m30u;
  const uint32_t m01=0x01010000u;
  int fd=ctz32((window1<<4)&m01);
  int after=32-fd;
  uint32_t segment1 = window1 << (after&31);
  uint64_t w2a = window12 >> (fd&63);
  int sd=ctz32((((uint32_t)w2a)<<4)&m01);
  uint32_t segment2 = ((uint32_t)w2a) << ((32-sd)&31);
  int lz=clz32((window4&0x00101010u)|0x10u);
  uint32_t mask=(uint32_t)(0x00000007FFFFFFFFull >> (lz&63));
  uint32_t segment4 = window4 & ~mask;
  long w3_idx = count-4-(lz>>3);
  uint32_t window3 = ld32(s,w3_idx) ^ (uint32_t)m30;
  long sdFromStart = fd+sd;
  long w3bits = w3_idx<<3;
  long extra = sdFromStart - w3bits;
  uint32_t segment3 = window3 & (0xFFFFFFFFu << (extra&31));
  uint64_t seg12=((uint64_t)segment1|((uint64_t)segment2<<32))^0x1E0000001E000000ull;
  uint64_t seg34=((uint64_t)segment3|((uint64_t)segment4<<24))^0x000000001E000000ull;
  const uint64_t m7f76=0x7F7676767F767676ull;
  uint64_t inv=((seg12+m7f76)|seg12)|((seg34+m7f76)|seg34);
  const uint64_t m7dfa=0x7DFAFA007DFAFA00ull;
  uint64_t o12=__builtin_bswap64(seg12)+m7dfa, o34=__builtin_bswap64(seg34)+m7dfa;
  uint64_t invalid=(inv&0x8080808080808080ull)|((o12|o34)&0x8000000080000000ull);
  if (invalid) return 0;
  if ((unsigned long)extra > 16) return 0;
  uint64_t v12=seg12*0x00640A01ull, v34=seg34*0x00640A01ull;
  uint64_t il=((v12>>16)&0x000000FF000000FFull)|(v34&0x00FF000000FF0000ull);
  uint32_t le=(uint32_t)(il|(il>>24)); /* byte0=a .. byte3=d */
  *out=__builtin_bswap32(le);
  return 1;
}
/* reference: 4 segments of 1-3 ASCII digits, each <=255, leading zeros allowed, separated by exactly 3 dots */
static inline int ref_parse(const uint8_t *s, long n, uint32_t *out){
  uint32_t a=0; int seg=0; long i=0;
  for(;;){ int d=0, v=0; while(i<n && s[i]>='0'&&s[i]<='9'){ v=v*10+(s[i]-'0'); d++; i++; if(d>3) return 0; }
    if(d==0||v>255) return 0; a=(a<<8)|v; seg++;
    if(seg==4) break; if(i>=n||s[i]!='.') return 0; i++; }
  if(i!=n) return 0; *out=a; return 1;
}
