# PopTree, measured

Experiment code for the maybenull.com post on
[Dynamically optimal weight-balanced binary search trees](https://arxiv.org/abs/2610.29574)
(Iacono & Pareek, 2026). The paper has no code or experiments; this is an independent
implementation of its SkipPopList (Section 3.1) from the text.

Question: the paper proves that the best possible self-adjusting weight-balanced (BB[α]) tree
costs OPT(X) up to a constant. Equation (1) charges a repeat search of x about log(n/d), d = the
distance to the nearest other key searched since x's previous search, plus an end-of-range term. How big is that effect, and how does PopTree
compare with a plain balanced tree and a splay tree, in one unit (BST nodes touched per search)?

## Files

| File | What |
|---|---|
| `poptree.cpp` | Everything: SkipPopList (clearing demotions, promotion, bottom-up gap repair), its conversion to a BST (two layouts), an invariant checker, a bottom-up splay tree, a static balanced BST, an offline entropy (frequency-bisection) BST, the OPT(X) terms, and the workloads. |
| `run.sh` | Invariant check at n = 3^8 (every invariant after every search), then the grid: n = 3^10 with m = 2,000,000 searches (5 seeds) and n = 3^12 with m = 20,000,000 (3 seeds). |
| `ablate_fill.sh` | Ablation of the one free choice in gap repair (which child to promote). |
| `analyze.py` | Writes `results/summary.json` and prints the markdown tables. |
| `chart_data.py` | Writes `figs/*.json`, the data behind the post's interactive charts. |
| `results/` | One JSON line per run, `summary.json`, `hardware.txt`. |

## Run

```sh
./run.sh            # ~10 min on 4 cores (2.1 GHz Xeon)
./ablate_fill.sh    # ~1 min
python3 analyze.py && python3 chart_data.py
```

Single run: `g++ -O2 -std=c++17 -o poptree poptree.cpp && ./poptree 10 2000000 stride81 1`
(`n = 3^10`, `m` searches, workload, seed; append `check` to verify every invariant after every
search, which is slow above n = 3^8).

## Cost model

- **Nodes touched per search** = depth + 1 of the searched key in the BST *before* the search,
  for every structure. Restructuring work is not included in it; for PopTree it is reported
  separately as skip-list promotions + demotions per search (`restructure_ops`). Splay rotations
  are not counted either.
- **PopTree's BST**: the single level-1 key is the root. Between two consecutive keys of level ℓ,
  the keys that first appear at level ℓ+1 form a group (at most 4 keys). `poptree` lays each group
  out as a right-leaning chain, smallest key on top, as in the paper's Appendix B.
  `poptree_median` lays it out as a median-split balanced BST instead. `poptree_level` is the
  skip-list level of the searched key, which does not depend on the layout. `min_alpha` /
  `min_alpha_median` are the smallest w(child)/w(parent) observed in each BST (w = size + 1).
- **splay**: rotations (depth − 1 per search) are not counted, only nodes touched.
- Only the second half of each sequence is measured (first half = warm-up); m ≈ 2·n·log2 n, as
  the theorem needs m = Ω(n log n).
- **OPT(X)** is evaluated as written in the paper's equation (1), in bits, with its unknown
  constant factor dropped. For a repeat search of x, the first term is log2(max(1, n/d)), d =
  distance to the nearest other key searched since x's previous search (segment tree over
  last-search times).

## Implementation choices the paper leaves open

- Gap repair promotes the occupant of one child bucket; the paper allows any. Default: the most
  recently searched child. `POPTREE_FILL=first` takes the leftmost occupied child instead. With
  the paper's chain layout it changes PopTree's cost by at most 0.06 nodes on the workloads in
  `ablate_fill.sh`; with the median layout it changes the ascending scan a lot (21.45 → 16.67).
- Initial structure: at every level the buckets ≡ 1 (mod 3) are occupied (a valid SkipPopList).
- "Sibling" buckets are the two adjacent buckets, as used in the proofs of Lemmas 20 and 23.

## Workloads (n = 3^10 unless noted)

uniform; Zipf(1) over ranks mapped to shuffled keys (`zipf_spread`) or to adjacent keys starting
at a random offset (`zipf_contig`); 729 hot keys (3^(L-4)) chosen at random (`hot_spread`),
adjacent (`hot_contig`), or evenly spaced `S` apart (`strideS`), uniform among them; random walk
with steps ±1..±8 (`walk`); repeated sequential scan, ascending (`sequential`) and descending
(`sequential_desc`).
