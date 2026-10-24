#!/bin/bash
# Full pipeline, ~80 min on 4 cores (4 single-threaded runs at a time).
set -e
cd "$(dirname "$0")"
mkdir -p results
[ -f data/input.txt ] || curl -sSL -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python3 filter_curves.py      # 1 s
./run_sweep.sh                # 49 runs, ~37 min
./run_seeds.sh                # 40 x 600-step + 8 x 2000-step runs, ~41 min
python3 analyze.py
python3 spectra_share.py
python3 chart_data.py
