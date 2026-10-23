#!/usr/bin/env bash
# Everything, in order. ~50 min on a 4-core CPU.
set -e
cd "$(dirname "$0")"
export PY=${PY:-python}
$PY cliff.py          # ~6 s
$PY cliff_k.py        # ~20 s
./run_grok.sh         # ~33 min (skips runs whose JSON already exists)
./run_lr1e-3.sh       # ~3 min
./run_digits.sh       # ~12 min
$PY bench_speed.py    # ~40 s, uses 4 threads; run on an idle machine
$PY analyze.py
$PY chart_data.py
