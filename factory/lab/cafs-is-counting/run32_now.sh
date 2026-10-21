#!/bin/sh
# one-off: run the int32 half after the in-flight uint64 run finishes
cd "$(dirname "$0")"
while pgrep -f "run.sh" >/dev/null || pgrep -x bench >/dev/null; do sleep 10; done
KS="2 8 16 64 256 1000 4096 16384 65536 262144 1000000"
OUT=results/grid_bench32.csv
echo "dist,N,K,algo,rep,ns_per_elem,ok,key_bits" > $OUT
for N in 1000000 10000000; do for D in uniform zipf dense; do taskset -c 2 ./bench32 $D $N 5 $KS >> $OUT; done; done
