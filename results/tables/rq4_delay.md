| Model | AUPRC oracle labels | AUPRC delayed labels | Relative loss | 95% CI of loss |
|---|---|---|---|---|
| Logistic regression | 0.338 ± 0.087 | 0.280 ± 0.050 | 0.15 ± 0.15 | [-0.03, 0.33] |
| LightGBM (tuned) | 0.698 ± 0.068 | 0.340 ± 0.204 | 0.53 ± 0.25 | [0.22, 0.84] |
| GRU | 0.544 ± 0.063 | 0.123 ± 0.126 | 0.78 ± 0.21 | [0.51, 1.04] |
| LightGBM + anomaly scores | 0.683 ± 0.079 | 0.167 ± 0.126 | 0.77 ± 0.15 | [0.58, 0.96] |
