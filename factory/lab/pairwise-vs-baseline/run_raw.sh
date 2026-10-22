#!/bin/sh
# MSE on raw ages (no target standardization), N=100, 5 seeds. ~8 min on 4 cores.
cd "$(dirname "$0")"
for s in 0 1 2 3 4; do echo "--arm mse_raw --seed $s"; done | xargs -P 4 -I{} sh -c 'python3 exp.py --n 100 --steps 3000 --eval_every 500 {} > logs/raw$(echo "{}" | tr -d " -/").log 2>&1'
