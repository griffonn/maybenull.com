#!/bin/bash
# Seeds 1-4 at each arm's best sweep LR (600 steps), plus 2000-step runs for 4 arms x 2 seeds.
cd "$(dirname "$0")"; mkdir -p results/seeds results/long logs
{
for arm in ns3 ns5 polar damped5; do for s in 0 1; do echo "$arm 0.02 $s 2000 results/long"; done; done
for spec in adamw:0.004 ns1:0.04 ns2:0.02 ns3:0.02 ns5:0.02 ns8:0.02 polar:0.02 damped1:0.04 damped3:0.02 damped5:0.02; do
  for s in 1 2 3 4; do echo "${spec%%:*} ${spec##*:} $s 600 results/seeds"; done; done
} | xargs -P4 -L1 bash -c 'THREADS=1 python3 muon_ns.py --arm $0 --lr $1 --seed $2 --steps $3 --out $4 --spectra >> logs/seeds.log 2>&1'
