#!/bin/sh
# Full grid, 4 single-threaded jobs at a time. ~80 min on 4 Xeon cores.
# Needs data/utk64.npz (python3 prep.py <UTKFace dir> data/utk64.npz) and results/baselines.json (python3 baselines.py).
cd "$(dirname "$0")"
{
  for s in 0 1 2; do echo "--arm pair --n 1000 --seed $s"; done
  echo "--arm pair --n 100 --seed 0 --steps 10000 --eval_every 1000 --out results/long"
  for s in 0 1 2 3 4; do echo "--arm pair --n 300 --seed $s"; done
  for s in 0 1 2 3 4; do echo "--arm pair --n 100 --seed $s"; done
  for s in 0 1 2; do echo "--arm pair --n 100 --seed $s --sampling natural"; done
  for n in 100 300; do for s in 0 1 2 3 4; do echo "--arm mse --n $n --seed $s"; done; done
  for s in 0 1 2; do echo "--arm mse --n 1000 --seed $s"; done
  for s in 0 1 2 3 4; do echo "--arm l1 --n 100 --seed $s"; done
  for s in 0 1 2; do echo "--arm mse --n 100 --seed $s --sampling natural"; done
} | xargs -P 4 -I{} sh -c 'python3 exp.py --steps 3000 --eval_every 500 {} > logs/$(echo "{}" | tr -d " -/" ).log 2>&1'
