| Hyp. | Ring share | A | B | Mean Δ AUPRC | 95% CI | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| H3a | 0% | GraphSAGE | LightGBM + graph features | -0.293 | [-0.404, -0.183] | 0.007 | LightGBM + graph features beats GraphSAGE |
| H3a | 25% | GraphSAGE | LightGBM + graph features | -0.294 | [-0.350, -0.239] | 0.001 | LightGBM + graph features beats GraphSAGE |
| H3a | 50% | GraphSAGE | LightGBM + graph features | -0.345 | [-0.620, -0.070] | 0.051 | inconclusive |
| H3a | 75% | GraphSAGE | LightGBM + graph features | -0.252 | [-0.572, +0.069] | 0.095 | inconclusive |
| H3b | 50% | LightGBM + graph features | LightGBM (tuned) | +0.063 | [+0.027, +0.099] | 0.025 | LightGBM + graph features beats LightGBM (tuned) |
| H3b | 75% | LightGBM + graph features | LightGBM (tuned) | +0.051 | [+0.037, +0.066] | 0.003 | LightGBM + graph features beats LightGBM (tuned) |
