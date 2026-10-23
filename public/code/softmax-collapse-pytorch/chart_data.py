"""Chart data for the post's interactive charts, from results/. Writes figs/*.json
(the post serves them from /lab/softmax-collapse-pytorch/)."""
import csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# 1. Target-logit gradient vs margin, K=32, per dtype (F.cross_entropy) + fixed fp32 + exact.
rows = list(csv.DictReader(open(os.path.join(R, "cliff.csv"))))
series = {}
for name, path, dt in [("F.cross_entropy, fp32", "F.cross_entropy", "torch.float32"),
                       ("F.cross_entropy, fp16", "F.cross_entropy", "torch.float16"),
                       ("F.cross_entropy, bf16", "F.cross_entropy", "torch.bfloat16"),
                       ("fixed CE, fp32", "fixed", "torch.float32")]:
    sub = [r for r in rows if r["path"] == path and r["dtype"] == dt and float(r["m"]) <= 40]
    m = [round(float(r["m"]), 2) for r in sub][::2]
    g = [abs(float(r["g_t"])) for r in sub][::2]
    series[name] = {"m": m, "g": [float("%.4g" % v) if v > 0 else None for v in g]}
exact = [r for r in rows if r["path"] == "fixed" and r["dtype"] == "torch.float64" and float(r["m"]) <= 40][::2]
series["exact"] = {"m": [round(float(r["m"]), 2) for r in exact], "g": [float("%.4g" % abs(float(r["true_g_t"]))) for r in exact]}
json.dump(series, open(os.path.join(OUT, "cliff.json"), "w"))

# 2. Training curves, one panel per arm group: exact train CE and test accuracy, every seed.
curves = {}
SHOW = ["std_fp32", "fixed_fp32", "std_fp32_eps1e-7", "std_bf16logits", "fixed_bf16logits"]
for f in sorted(glob.glob(os.path.join(R, "grok", "*.json"))):
    d = json.load(open(f))
    if d["arm"] not in SHOW:
        continue
    L = d["log"][::5]  # every 50 steps
    curves.setdefault(d["arm"], []).append({"seed": d["seed"], "step": [r[0] for r in L],
        "ce": [float("%.3g" % max(r[1], 1e-12)) for r in L], "margin": [round(r[3], 2) for r in L],
        "test": [round(100 * r[7], 1) for r in L]})
json.dump(curves, open(os.path.join(OUT, "grok-curves.json"), "w"))
print("wrote figs/cliff.json, figs/grok-curves.json")
