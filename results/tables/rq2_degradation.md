| Model (static) | Type | AUPRC pre-drift (d50–54) | AUPRC post-drift (d54–90) | Degradation D | 95% CI of D |
|---|---|---|---|---|---|
| Logistic regression | supervised | 0.713 ± 0.280 | 0.042 ± 0.025 | 0.94 ± 0.03 | [0.90, 0.97] |
| LightGBM (tuned) | supervised | 0.909 ± 0.112 | 0.107 ± 0.046 | 0.88 ± 0.06 | [0.81, 0.95] |
| GRU | supervised | 0.875 ± 0.134 | 0.067 ± 0.040 | 0.92 ± 0.04 | [0.87, 0.98] |
| Isolation Forest | unsupervised | 0.197 ± 0.211 | 0.207 ± 0.021 | -2.05 ± 3.46 | [-6.34, 2.24] |
| Autoencoder | unsupervised | 0.031 ± 0.015 | 0.028 ± 0.005 | -0.56 ± 1.90 | [-2.91, 1.80] |
