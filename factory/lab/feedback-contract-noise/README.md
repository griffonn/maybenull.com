# Release rules for an ML agent's validation score

Experiment code for the maybenull.com post on
[An Explicit Feedback Contract for Iterative ML-Agent Harnesses](https://arxiv.org/abs/2610.28798)
(Yao et al., October 2026). The paper's release rule (Algorithm 1) shows a new validation score only
when a submission beats the current champion by one paired standard error. It is the
"parameter-free" variant of Blum & Hardt's
[Ladder](https://proceedings.mlr.press/v37/blum15.html) (ICML 2015).

Question: how much fake validation gain does that rule let through, as a function of how much each
submission changes, and what does it cost on a task with real signal? Compared against exact
feedback and a fixed-margin Ladder (η = 0.005). CPU only, numpy + scikit-learn.

## Files

| File | What |
|---|---|
| `sim.py` | Pure-noise simulations, n = 25,000 random labels. (a) local searcher: each submission is the champion with m prediction bits flipped, m from 1 to 10,000, 1,000 submissions, 5 seeds. (b) Blum & Hardt's boosting attack: 1,000 random vectors, then the majority vote of the ones the harness called good. (c) first-step pass rate of the paired-SE rule. ~1 min. |
| `agent.py` | Scripted hill-climber with a HistGradientBoostingClassifier, 20k train / 25k validation / 50k test rows of `make_classification` (30 features, 10 informative, 30% label noise); task `null` replaces the labels with coin flips. Searcher `tweak` changes one hyperparameter of the champion per submission; searcher `ensemble` adds one new member (one tweak, fresh seed) to the champion's probability-averaged ensemble. 200 submissions, no stopping; patience is applied afterwards in `analyze.py`. |
| `run_agent.sh` | `./run_agent.sh tweak` (~10 min on 4 cores) and `./run_agent.sh ensemble` (~25 min). 2 tasks × 3 rules × 5 seeds each. |
| `analyze.py` | Tables to stdout, everything to `results/summary.json`. Gains are measured from the first submitted model; "inflation" = validation gain − fresh-test gain. Patience p = stop after p consecutive submissions with no update. |
| `plot.py` | `figs/noise-sweep.png`. |
| `results/` | One JSON per agent run (per-submission log: knob, predictions changed m, z, accepted, champion val/test error), simulation CSVs, `summary.json`, `hardware.txt`. |

## Rules

- `exact`: champion = any submission with lower validation error.
- `pse`: accept if error < champion error − std(paired loss difference)/√n, `np.std` with ddof = 0,
  as Algorithm 1 writes it (`pse1` in `sim.py` uses ddof = 1). The released score is rounded to 1/n,
  which for 0-1 loss changes nothing.
- `ladder`: accept if error < champion error − 0.005.

## Rerun

```sh
pip install -r requirements.txt
python sim.py                    # ~1 min
./run_agent.sh tweak             # ~10 min on 4 cores
./run_agent.sh ensemble          # ~25 min on 4 cores
python analyze.py && python plot.py
```

Hardware: 4-core Intel Xeon @ 2.1 GHz, 15 GB RAM, no GPU (`results/hardware.txt`). Everything is
seeded; reruns on the same package versions reproduce the numbers exactly.

## Not done

No LLM agent (the paper's agents are Codex and Claude; ours are scripted). No AUC or NDCG, only
0-1 loss, where per-example paired differences are well defined. Validation size fixed at 25,000.
