# TTT analysis (400 puzzles)

## Fixed budgets

| method | budget | arc_score | cell_accuracy | size_match | seconds_per_puzzle |
|---|---|---|---|---|---|
| vote | 0 | 0.031 | 0.520 | 0.680 | 19.307 |
| vote | 8 | 0.074 | 0.496 | 0.635 | 24.645 |
| vote | 16 | 0.100 | 0.498 | 0.624 | 25.050 |
| vote | 32 | 0.124 | 0.558 | 0.681 | 27.980 |
| vote | 64 | 0.134 | 0.540 | 0.655 | 37.026 |
| poe | 0 | 0.045 | 0.602 | 0.779 | 23.665 |
| poe | 8 | 0.086 | 0.556 | 0.705 | 28.356 |
| poe | 16 | 0.109 | 0.611 | 0.755 | 29.159 |
| poe | 32 | 0.155 | 0.644 | 0.771 | 32.689 |
| poe | 64 | 0.166 | 0.658 | 0.779 | 42.134 |


## Adaptive stopping (offline)

| method | rule | arc_score | seconds_per_puzzle | compute_vs_max_budget | mean_stop_step |
|---|---|---|---|---|---|
| vote | loo_loss < 0.00838 (q10) | 0.124 | 36.880 | 0.996 | 59.460 |
| poe | loo_loss < 0.00838 (q10) | 0.169 | 42.009 | 0.997 | 59.460 |
| vote | loo_loss < 0.0273 (q25) | 0.120 | 34.994 | 0.945 | 53.140 |
| poe | loo_loss < 0.0273 (q25) | 0.152 | 40.051 | 0.951 | 53.140 |
| vote | loo_loss < 0.0685 (q50) | 0.077 | 29.921 | 0.808 | 37.760 |
| poe | loo_loss < 0.0685 (q50) | 0.106 | 34.530 | 0.820 | 37.760 |
| vote | loo_loss < 0.166 (q75) | 0.074 | 24.797 | 0.670 | 19.120 |
| poe | loo_loss < 0.166 (q75) | 0.081 | 29.131 | 0.691 | 19.120 |
| vote | loo_loss plateau (<2% drop) | 0.135 | 35.130 | 0.949 | 49.680 |
| poe | loo_loss plateau (<2% drop) | 0.161 | 40.065 | 0.951 | 49.680 |
| vote | loo_loss plateau (<5% drop) | 0.138 | 34.255 | 0.925 | 46.740 |
| poe | loo_loss plateau (<5% drop) | 0.163 | 39.110 | 0.928 | 46.740 |
| vote | loo_loss plateau (<10% drop) | 0.128 | 32.464 | 0.877 | 40.400 |
| poe | loo_loss plateau (<10% drop) | 0.147 | 37.173 | 0.882 | 40.400 |


## Does the signal predict success? (AUC, 0.5 = no)

| step | signal | auc | n_solved | n_unsolved |
|---|---|---|---|---|
| 0 | loo_loss | 0.581 | 13 | 387 |
| 8 | loo_loss | 0.637 | 31 | 369 |
| 16 | loo_loss | 0.710 | 41 | 359 |
| 32 | loo_loss | 0.690 | 52 | 348 |
| 64 | loo_loss | 0.764 | 55 | 345 |
