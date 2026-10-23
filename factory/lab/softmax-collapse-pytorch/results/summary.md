| arm | final test acc % (mean ± sd, min) | seeds reaching 99% | steps to 99% (mean) | loss spikes / run | worst test-acc dip, pts | final mean margin |
|---|---|---|---|---|---|---|
| std_fp32 | 98.9 ± 1.9 (min 95.6) | 3/5 | 4747 | 7.0 (max 7) | 15.4 | 9.4 |
| fixed_fp32 | 100.0 ± 0.0 (min 100.0) | 5/5 | 6766 | 0.4 (max 1) | 1.9 | 17.0 |
| std_fp32_eps1e-7 | 99.9 ± 0.2 (min 99.6) | 5/5 | 6574 | 0.0 (max 0) | 0.4 | 14.7 |
| std_fp32_ls0.01 | 46.6 ± 3.8 (min 42.2) | 0/5 | — | 0.0 (max 0) | 0.0 | 4.6 |
| std_fp32_loss64 | 98.1 ± 3.6 (min 91.7) | 4/5 | 5670 | 0.2 (max 1) | 22.2 | 17.2 |
| std_autocast_bf16 | 98.9 ± 1.6 (min 96.9) | 3/5 | 5547 | 6.2 (max 7) | 24.3 | 13.9 |
| std_bf16logits | 97.2 ± 3.1 (min 92.7) | 4/5 | 5928 | 40.8 (max 46) | 50.9 | 4.3 |
| fixed_bf16logits | 99.2 ± 1.2 (min 97.3) | 4/5 | 5560 | 0.0 (max 0) | 1.5 | 17.2 |
| std_bf16 | 46.2 ± 3.5 (min 42.4) | 0/5 | — | 30.8 (max 32) | 0.0 | 5.5 |
| fixed_bf16 | 44.3 ± 2.9 (min 41.4) | 0/5 | — | 8.0 (max 13) | 0.0 | 14.3 |
fixed_wd0.0 {'final_test': (0.9713, 0.9667, 0.9778), 'final_median_margin': (31.9556, 31.7427, 32.3568), 'final_min_margin': (16.9272, 16.9029, 16.9735), 'max_dead': (1.0, 1.0, 1.0), 'first_dead_step': (183.3333, 150, 200), 'spikes': (0, 0, 0), 'n': 3}
fixed_wd0.01 {'final_test': (0.9713, 0.9639, 0.9778), 'final_median_margin': (29.1854, 28.9012, 29.4513), 'final_min_margin': (16.1129, 15.9549, 16.2309), 'max_dead': (0.9986, 0.9979, 0.9993), 'first_dead_step': (183.3333, 150, 200), 'spikes': (0, 0, 0), 'n': 3}
std_wd0.0 {'final_test': (0.9685, 0.9611, 0.9722), 'final_median_margin': (29.0585, 28.798, 29.1953), 'final_min_margin': (15.0438, 14.2822, 15.4764), 'max_dead': (0.971, 0.9694, 0.9736), 'first_dead_step': (183.3333, 150, 200), 'spikes': (0, 0, 0), 'n': 3}
std_wd0.01 {'final_test': (0.9722, 0.9667, 0.9778), 'final_median_margin': (27.3672, 26.9988, 27.6938), 'final_min_margin': (14.9351, 14.8682, 14.9864), 'max_dead': (0.9506, 0.9471, 0.9527), 'first_dead_step': (183.3333, 150, 200), 'spikes': (0, 0, 0), 'n': 3}
