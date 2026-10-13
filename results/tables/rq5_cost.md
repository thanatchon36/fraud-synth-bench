| Model (0.5% fraud) | AUPRC | Recovered $ / investigator-hour (rank by p) | … (rank by p × amount) | Total cost $ (missed + 6.67/alert) |
|---|---|---|---|---|
| LightGBM (tuned) | 0.738 ± 0.066 | 2507 ± 1178 | 3167 ± 1440 | 42688 ± 18342 |
| LightGBM (default) | 0.620 ± 0.106 | 2022 ± 1202 | 2987 ± 1649 | 55627 ± 20243 |
| Random forest | 0.726 ± 0.046 | 2498 ± 1343 | 3141 ± 1597 | 42943 ± 15645 |
| Logistic regression | 0.326 ± 0.068 | 1161 ± 446 | 1885 ± 809 | 78595 ± 35720 |
| GRU | 0.552 ± 0.091 | 1101 ± 544 | 2100 ± 762 | 80178 ± 36308 |
| FT-Transformer | 0.488 ± 0.071 | 1384 ± 393 | 2281 ± 952 | 72644 ± 36157 |
| Isolation Forest | 0.189 ± 0.079 | 1408 ± 776 | n/a (score) | 71996 ± 23629 |
| Autoencoder | 0.032 ± 0.011 | 57 ± 86 | n/a (score) | 108019 ± 45092 |
