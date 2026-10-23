#!/usr/bin/env bash
# The paper's lr (1e-3) on our model: std and fixed CE, seeds 0-1. ~3 min on 4 cores.
cd "$(dirname "$0")"; PY=${PY:-python}; mkdir -p results/grok_lr1e-3
for a in std_fp32 fixed_fp32; do for s in 0 1; do
  LR=1e-3 OUT=results/grok_lr1e-3/${a}_s${s}.json $PY grok.py $a $s &
done; done; wait
