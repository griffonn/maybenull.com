#!/usr/bin/env bash
# Full grid: 10 arms x 5 seeds, 4 runs in parallel (one thread each). ~30 min on 4 cores.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p results/grok logs
for arm in std_fp32 fixed_fp32 std_fp32_eps1e-7 std_fp32_ls0.01 std_fp32_loss64 std_bf16 fixed_bf16 std_autocast_bf16 std_bf16logits fixed_bf16logits; do
  for s in 0 1 2 3 4; do echo "$arm $s"; done
done | while read a s; do [ -f results/grok/${a}_s${s}.json ] || echo "$a $s"; done | xargs -P 4 -n 2 $PY grok.py 2>&1 | tee -a logs/grok.log
