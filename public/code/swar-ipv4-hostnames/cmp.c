#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
int main(){ char l[256]; while(fgets(l,256,stdin)){ l[strcspn(l,"\n")]=0; struct in_addr a; char o[32];
  int p=inet_pton(AF_INET,l,&a); printf("%-18s pton=%-16s", l, p==1?inet_ntop(AF_INET,&a,o,32):"REJECT");
  int q=inet_aton(l,&a); printf(" aton=%s\n", q?inet_ntop(AF_INET,&a,o,32):"REJECT"); } }
