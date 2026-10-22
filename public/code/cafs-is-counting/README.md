# CAFS vs. counting sort

Experiment code for the maybenull.com post on CAFS
([arXiv 2605.25040](https://arxiv.org/abs/2605.25040), code
[kexibq-official/cafs-lib](https://github.com/kexibq-official/cafs-lib) @ `13488afef446`).

Question: CAFS is a hash-count sort (count each distinct value, sort the K keys,
write the runs back). Its paper compares it only against general-purpose sorts.
How does it compare with an ordinary hash-map count, and how much of it is the
AVX2 cache-line bucket?

## Files

| File | What |
|---|---|
| `bench.cpp` | 7 contenders × input generator. `-DKEY32` switches keys from `uint64_t` to `int32_t`. |
| `ablate.cpp` | `cafs_hashmap`: CAFS with only its bucket table swapped for `ankerl::unordered_dense`. |
| `diag.cpp` | Re-runs CAFS's own Chao1 estimator and table sizing; counts elements that spill out of the bucket table. |
| `analyze.py` | Prints median tables from `results/grid*.csv`. |
| `plot.py` | Writes `figs/*.png`. Needs matplotlib. |
| `chart_data.py` | Writes `figs/*.json`, the data behind the interactive charts in the post. |
| `third_party/` | Vendored headers: CAFS (`cafs.hpp`, `cafs2.hpp`), pdqsort, ska_sort, ankerl unordered_dense. Commits in `SOURCES.txt`. |
| `results/` | Raw CSVs (one row per run), `diag.csv`, `hardware.txt`. |

Contenders: `cafs` (with pdqsort as its fallback), `std_sort`, `pdqsort`,
`ska_sort` (radix), and three counting sorts that share one 4-line emit step
(sort the (key, count) pairs, `fill_n` each run): `count_std_umap`
(`std::unordered_map`), `count_ankerl` (`ankerl::unordered_dense::map`),
`count_flat` (a 30-line linear-probing table).

Inputs: `uniform` (K random 64-bit keys, uniform frequencies), `zipf` (s = 1 over
K keys), `dense` (keys 0..K−1), `runs` (uniform keys in runs of geometric length,
mean 16). Every result is checked: rep 0 against `std::sort`, later reps with
`is_sorted` (grid) or against `std::sort` (ablation).

## Run

```sh
./run.sh              # full grid + ablation, ~45 min, writes results/
python3 analyze.py 64 # tables (or 32)
python3 plot.py       # figures
g++ -O2 -march=x86-64-v3 -std=c++20 -o diag diag.cpp && ./diag zipf 10000000 1000 65536 1000000
```

Hardware: 4-vCPU KVM guest, Intel Xeon @ 2.1 GHz (AVX2 + AVX-512, 2 MiB L2 per
core, 260 MiB shared L3), 15 GB RAM, Linux 6.18, g++ 13.3 `-O3 -march=x86-64-v3`.
No frequency pinning available (VM); every benchmark ran pinned to one core with
nothing else running, 5 repetitions per cell, medians reported. Wall-clock for the
whole set: about 45 minutes.

Note: the `uint64` grid (`results/grid_bench.csv`) was produced before the
`key_bits` column and the `runs` distribution were added to `bench.cpp`; its
`key_bits` column (all 64) was appended afterwards, and the measured code path is
unchanged. `grid_runs_*.csv` hold the `runs` rows. `REP=n ./diag ...` diagnoses
the input of repetition n.
