#!/bin/sh
# 10k-step runs at N=100 for both arms, 3 seeds (pair seed 0 comes from run.sh). ~25 min on 4 cores.
cd "$(dirname "$0")"
{
  for s in 1 2; do echo "--arm pair --seed $s"; done
  for s in 0 1 2; do echo "--arm mse --seed $s"; done
} | xargs -P 4 -I{} sh -c 'python3 exp.py --n 100 --steps 10000 --eval_every 1000 --out results/long {} > logs/long$(echo "{}" | tr -d " -/").log 2>&1'
