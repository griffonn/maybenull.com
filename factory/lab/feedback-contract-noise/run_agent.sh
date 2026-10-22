#!/bin/sh
# usage: ./run_agent.sh tweak|ensemble
# 2 tasks x 3 rules x 5 seeds x 200 submissions, 4 single-threaded workers.
# tweak: ~10 min on 4 cores (repeat proposals are cached); ensemble: ~25 min.
S=${1:-tweak}
mkdir -p results logs
for task in null signal; do for rule in exact pse ladder; do for seed in 0 1 2 3 4; do
  echo "$task $rule $seed"; done; done; done |
  OMP_NUM_THREADS=1 xargs -P 4 -L 1 sh -c 'python agent.py $0 $1 $2 200 '$S' > logs/agent_'$S'_$0_$1_$2.log 2>&1'
