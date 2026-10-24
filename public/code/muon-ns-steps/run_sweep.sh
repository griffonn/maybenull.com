#!/bin/bash
# LR sweep, seed 0, 600 steps, 4 runs in parallel (1 thread each).
cd "$(dirname "$0")"; mkdir -p results/sweep logs
{
for arm in ns1 ns2 ns3 ns5 ns8 polar damped1 damped3 damped5; do
  for lr in 0.01 0.02 0.04 0.08 0.16; do echo "$arm $lr"; done; done
for lr in 0.001 0.002 0.004 0.008; do echo "adamw $lr"; done
} | xargs -P4 -L1 bash -c 'THREADS=1 python3 muon_ns.py --arm $0 --lr $1 --seed 0 --steps 600 --out results/sweep >> logs/sweep.log 2>&1'
