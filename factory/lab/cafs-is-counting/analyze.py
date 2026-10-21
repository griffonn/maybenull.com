import csv, statistics as st, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "results/grid.csv")))
assert all(r["ok"] == "1" for r in rows), "a sort produced wrong output"
d = defaultdict(list)
for r in rows: d[(r["dist"], int(r["N"]), int(r["K"]), r["algo"])].append(float(r["ns_per_elem"]))
algos = ["cafs","count_ankerl","count_flat","count_std_umap","pdqsort","ska_sort","std_sort"]
for dist in ["uniform","zipf","dense"]:
  for N in sorted({k[1] for k in d}):
    print(f"\n{dist} N={N:.0e}  median ns/elem over 5 reps (min-max for cafs & ankerl)")
    print("K".rjust(8), *[a[:12].rjust(12) for a in algos], "ankerl/cafs".rjust(12))
    for K in sorted({k[2] for k in d if k[0]==dist and k[1]==N}):
      med = {a: st.median(d[(dist,N,K,a)]) for a in algos if (dist,N,K,a) in d}
      if len(med) < len(algos): continue
      print(str(K).rjust(8), *[f"{med[a]:12.2f}" for a in algos], f"{med['count_ankerl']/med['cafs']:12.2f}")
