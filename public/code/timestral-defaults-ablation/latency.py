"""Batch-1 latency, 48-step horizon, 4 threads, best of N."""
import pickle, time, numpy as np, torch
from timestral import TimestralForecaster

torch.set_num_threads(4)
s = pickle.load(open("bench.pkl", "rb"))["A"]["ercot"]["series"][0][:-48]
for m in ("5M", "28M", "123M", "209M"):
    f = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{m}")
    row = []
    for c in (2048, 8192, 32768):
        for sym in (False, True):
            x = s[-c:]
            f.forecast(x, 48, symmetric_averaging=sym)
            ts = []
            for _ in range(3):
                t = time.perf_counter(); f.forecast(x, 48, symmetric_averaging=sym); ts.append(time.perf_counter() - t)
            row.append(f"ctx{c}{'+mirror' if sym else ''}={1000 * min(ts):.0f}ms")
    print(m, " ".join(row), flush=True)
