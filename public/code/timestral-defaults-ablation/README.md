# Timestral defaults ablation: reproduction code

Code for the post [Timestral's defaults, measured](https://maybenull.com/posts/timestral-defaults-ablation/).

## Pinned versions

- Timestral code: `mistralai/timestral` at `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`
- Weights: `mistralai/Timestral-{5M,28M,123M,209M}` from the Hugging Face Hub
  (209M snapshot `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`)
- Data: `autogluon/chronos_datasets` at revision `eeecad0b82a8c237e212ce6f8d1abecb513e2cec`
- Python 3.11, packages in `requirements.txt`

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
./run_all.sh          # ~45 min on 4 vCPUs; writes analysis.txt and latency.txt
```

`analysis.txt` and `latency.txt` in this folder are the outputs behind the post. Accuracy numbers
are deterministic. Latencies vary by machine and by run (I saw about ±15% between runs on the same VM).

## Files

| file | what it does |
|---|---|
| `common.py` | dataset spec, the two forecast windows, metrics, and `combine()` which rebuilds `forecast()` output from the two raw passes |
| `prep.py` | downloads the 9 datasets and samples the series (seed 42) for both windows |
| `check_equivalence.py` | checks `combine()` reproduces `TimestralForecaster.forecast()` defaults (prints max abs diff 0.0) |
| `run.py MODEL CTX ORIGIN [--window W]` | runs the forward pass and the pass on `-context` and saves raw quantiles; `--window` swaps the scaler's cumulative statistics for a trailing window |
| `analyze.py` | prints every accuracy table in the post |
| `latency.py` | batch-1 latency table |

## The two forecast windows ("latest" and "earlier")

For each sampled series, the **latest** window is the last H steps of the data. The **earlier** window
is the same series with its last 7·m steps removed first (m = season length), i.e. one week earlier for the
hourly, 30-min and 15-min datasets, 49 days earlier for M5 (daily, m=7) and 30 days earlier for exchange
rates (daily, m=1). In the code these are origins `A` and `B`.
