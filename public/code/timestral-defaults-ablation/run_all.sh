#!/usr/bin/env bash
# Full reproduction: about 45 minutes on a 4-vCPU machine, CPU only.
set -euo pipefail
python prep.py
python check_equivalence.py
for m in 5M 28M 123M 209M; do python run.py $m 32768 A; done
for m in 5M 28M 209M; do for c in 512 2048 8192; do python run.py $m $c A; done; done
for m in 5M 209M; do python run.py $m 32768 A --window 2048; done
for m in 5M 209M; do for c in 2048 32768; do python run.py $m $c B; done; done
python analyze.py | tee analysis.txt
python latency.py | tee latency.txt
