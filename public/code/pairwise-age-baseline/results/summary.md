| arm | train sampling | N | seeds | test MAE (final) | MAE if checkpoint picked on test | pred SD / true SD |
|---|---|---|---|---|---|---|
| const_train_mean | balanced | 100 | 5 | 22.18 ± 0.26 (21.9–22.5) | nan | nan |
| l1 | balanced | 100 | 5 | 11.98 ± 0.68 (11.0–12.7) | 11.47 | 0.78 |
| mse | balanced | 100 | 5 | 12.71 ± 0.55 (12.3–13.6) | 12.52 | 0.73 |
| mse (10k steps) | balanced | 100 | 3 | 12.03 ± 0.60 (11.5–12.7) | 11.37 | 0.82 |
| mse_raw | balanced | 100 | 5 | 13.66 ± 1.14 (12.4–15.5) | 10.93 | 0.85 |
| pair | balanced | 100 | 5 | 13.74 ± 1.32 (12.1–15.1) | 13.25 | 1.03 |
| pair (10k steps) | balanced | 100 | 3 | 12.98 ± 0.32 (12.7–13.3) | 12.35 | 0.99 |
| ridge_pixels | balanced | 100 | 5 | 17.22 ± 1.39 (15.9–19.0) | nan | nan |
| const_train_mean | balanced | 300 | 5 | 22.19 ± 0.12 (22.1–22.4) | nan | nan |
| mse | balanced | 300 | 5 | 10.65 ± 0.70 (10.0–11.5) | 10.35 | 0.86 |
| pair | balanced | 300 | 5 | 12.68 ± 1.24 (11.2–14.1) | 12.28 | 1.07 |
| ridge_pixels | balanced | 300 | 5 | 15.50 ± 0.47 (14.8–16.0) | nan | nan |
| const_train_mean | balanced | 1000 | 5 | 22.10 ± 0.04 (22.1–22.2) | nan | nan |
| mse | balanced | 1000 | 3 | 8.53 ± 0.33 (8.2–8.8) | 8.46 | 0.94 |
| pair | balanced | 1000 | 3 | 10.29 ± 0.11 (10.2–10.4) | 10.20 | 1.11 |
| ridge_pixels | balanced | 1000 | 5 | 14.36 ± 0.28 (13.9–14.6) | nan | nan |
| const_train_mean | natural | 100 | 5 | 15.95 ± 0.52 (15.4–16.8) | nan | nan |
| mse | natural | 100 | 3 | 11.13 ± 0.83 (10.6–12.1) | 11.12 | 0.54 |
| pair | natural | 100 | 3 | 12.39 ± 1.54 (10.6–13.5) | 11.94 | 0.66 |
| ridge_pixels | natural | 100 | 5 | 13.99 ± 0.37 (13.6–14.4) | nan | nan |
| const_train_mean | natural | 300 | 5 | 15.78 ± 0.26 (15.6–16.2) | nan | nan |
| ridge_pixels | natural | 300 | 5 | 12.50 ± 0.27 (12.1–12.9) | nan | nan |
| const_train_mean | natural | 1000 | 5 | 15.69 ± 0.13 (15.5–15.9) | nan | nan |
| ridge_pixels | natural | 1000 | 5 | 11.51 ± 0.32 (11.1–11.9) | nan | nan |
constant (whole-dataset median) MAE on the test set: 15.21
