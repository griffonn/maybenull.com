#!/usr/bin/env bash
# Digits MLP: {std, fixed} x wd {0, 0.01} x 3 seeds, 4 in parallel. ~3 min on 4 cores.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p results/digits logs
for l in std fixed; do for wd in 0 0.01; do for s in 0 1 2; do echo "$l $wd $s"; done; done; done \
  | xargs -P 4 -n 3 $PY digits.py 2>&1 | tee -a logs/digits.log
