#!/usr/bin/env bash
# Full grid. ~10 min on 4 cores. Results: results/*.jsonl, hardware/time in results/hardware.txt
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
g++ -O2 -std=c++17 -o poptree poptree.cpp
{ echo "date: $(date -u)"; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2; g++ --version | head -1; } > results/hardware.txt
WL="uniform sequential sequential_desc walk hot_contig stride3 stride9 stride27 stride81 hot_spread zipf_contig zipf_spread"
start=$(date +%s)
# 1) invariant check: every invariant after every search, n = 3^8
for w in $WL; do echo "./poptree 8 60000 $w 7 check"; done | xargs -P 4 -I{} sh -c '{}' > results/check_L8.jsonl
# 2) main grid: n = 3^10, m = 2,000,000 (>= 2 n log2 n), 5 seeds
for s in 1 2 3 4 5; do for w in $WL; do echo "./poptree 10 2000000 $w $s"; done; done | xargs -P 4 -I{} sh -c '{}' > results/grid_L10.jsonl
# 3) larger n = 3^12, m = 20,000,000, 3 seeds
for s in 1 2 3; do for w in $WL; do echo "./poptree 12 20000000 $w $s"; done; done | xargs -P 4 -I{} sh -c '{}' > results/grid_L12.jsonl
echo "wall_clock_s: $(( $(date +%s) - start ))" >> results/hardware.txt
