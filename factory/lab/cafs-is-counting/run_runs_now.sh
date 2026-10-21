#!/bin/sh
# one-off: the 'runs' distribution, added after the main grid started
cd "$(dirname "$0")"
while pgrep -f "run32_now.sh" >/dev/null || pgrep -f "run.sh" >/dev/null; do sleep 10; done
KS="2 8 16 64 256 1000 4096 16384 65536 262144 1000000"
for B in bench_x bench32_x; do
  OUT=results/grid_runs_$B.csv
  echo "dist,N,K,algo,rep,ns_per_elem,ok,key_bits" > $OUT
  for N in 1000000 10000000; do taskset -c 2 ./$B runs $N 5 $KS >> $OUT; done
done
