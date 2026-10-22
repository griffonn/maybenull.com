#!/usr/bin/env bash
# Ablation: gap repair promotes the leftmost occupied child instead of the most recently searched one.
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o poptree poptree.cpp
for w in uniform zipf_spread hot_contig stride81 sequential; do echo "POPTREE_FILL=first ./poptree 10 2000000 $w 1"; echo "POPTREE_FILL=first ./poptree 8 60000 $w 7 check"; done | xargs -P 4 -I{} sh -c '{}' > results/ablate_fill.jsonl
