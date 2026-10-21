#!/bin/sh
cd "$(dirname "$0")"
while pgrep -f "_now.sh" | grep -v $$ >/dev/null; do sleep 10; done
KS="2 8 16 64 256 1000 4096 16384 65536 262144 1000000"
for B in ablate ablate32; do
  OUT=results/ablation_$B.csv
  echo "dist,N,K,algo,rep,ns_per_elem,ok,key_bits" > $OUT
  for D in uniform zipf runs; do taskset -c 2 ./$B $D 10000000 5 $KS >> $OUT; done
done
