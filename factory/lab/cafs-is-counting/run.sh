#!/bin/sh
# Full grid: uint64 and int32 keys, N in {1e6, 1e7}, 4 distributions, 11 K values, 5 reps.
# ~70 min on a 4-core Xeon @2.1GHz. Pinned to one core, one process at a time.
set -e
cd "$(dirname "$0")"
g++ -O3 -march=x86-64-v3 -std=c++20 -DNDEBUG -o bench bench.cpp
g++ -O3 -march=x86-64-v3 -std=c++20 -DNDEBUG -DKEY32 -o bench32 bench.cpp
mkdir -p results
KS="2 8 16 64 256 1000 4096 16384 65536 262144 1000000"
for B in bench bench32; do
  OUT=results/grid_$B.csv
  echo "dist,N,K,algo,rep,ns_per_elem,ok,key_bits" > $OUT
  for N in 1000000 10000000; do
    for D in uniform zipf dense runs; do
      taskset -c 2 ./$B $D $N 5 $KS >> $OUT
    done
  done
done

# Ablation: CAFS with only its bucket table swapped for ankerl::unordered_dense.
g++ -O3 -march=x86-64-v3 -std=c++20 -DNDEBUG -o ablate ablate.cpp
g++ -O3 -march=x86-64-v3 -std=c++20 -DNDEBUG -DKEY32 -o ablate32 ablate.cpp
for B in ablate ablate32; do
  OUT=results/ablation_$B.csv
  echo "dist,N,K,algo,rep,ns_per_elem,ok,key_bits" > $OUT
  for D in uniform zipf runs; do
    taskset -c 2 ./$B $D 10000000 5 $KS >> $OUT
  done
done
