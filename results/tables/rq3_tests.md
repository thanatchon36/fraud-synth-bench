| Hyp. | Ring share | A | B | Mean Δ AUPRC | 95% CI | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| H3a | 0% | GraphSAGE | LightGBM + graph features | -0.326 | [-0.495, -0.157] | 0.023 | LightGBM + graph features beats GraphSAGE |
| H3a | 25% | GraphSAGE | LightGBM + graph features | -0.293 | [-0.346, -0.239] | 0.001 | LightGBM + graph features beats GraphSAGE |
| H3a | 50% | GraphSAGE | LightGBM + graph features | -0.347 | [-0.622, -0.072] | 0.050 | LightGBM + graph features beats GraphSAGE |
| H3a | 75% | GraphSAGE | LightGBM + graph features | -0.251 | [-0.567, +0.065] | 0.092 | inconclusive |
| H3b | 50% | LightGBM + graph features | LightGBM (tuned) | +0.063 | [+0.027, +0.099] | 0.025 | LightGBM + graph features beats LightGBM (tuned) |
| H3b | 75% | LightGBM + graph features | LightGBM (tuned) | +0.051 | [+0.037, +0.066] | 0.003 | LightGBM + graph features beats LightGBM (tuned) |
