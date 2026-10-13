# Executive summary: what model to run against payment fraud, and what to worry about instead

*Generated from `results/summary.json` by `make paper`. Every number is from the reproducible run (fraudlab v0.1.0). **All data is synthetic.** Treat the figures as evidence about mechanisms, not forecasts for any real portfolio.*

## Bottom line

**Use a tuned gradient-boosted tree (LightGBM) with good history and relationship features, rank alerts by expected loss, and spend engineering effort on retraining speed and label latency rather than on newer model families.**

## What we found

| Question | Finding | So what |
|---|---|---|
| Do newer AI models beat the classic approach? | No. Tuned LightGBM scored 0.74 AUPRC at 0.5% fraud against 0.49 for a tabular transformer, which was also about 41 ms vs. 3 ms per 1,000 decisions. TabPFN wasn't evaluated (license). | No case here for replacing a well-tuned tree model. |
| Do graph neural networks catch fraud rings better? | No. LightGBM with simple network features (shared devices, fan-in, connected-component size) beat GraphSAGE. Those features added +0.06 AUPRC when half the fraud was ring-based. | Build relationship features, but a GNN isn't required to use them. |
| What happens when fraudsters change tactics? | Every supervised model lost 88%+ of its detection quality. Weekly refits on all labels recovered 26% of the gap in the following week, and close to the pre-change level two to three weeks later. Retraining that holds out the last 10 days for validation recovered **nothing** in that week. | Retraining cadence and pipeline design matter more than the choice of model. |
| How much do slow fraud confirmations hurt? | Chargeback-style labels arriving 7–30 days late cut the best model's detection quality by 53%. Adding unsupervised anomaly scores did not help. | Shortening label latency is worth real money. |
| Which model saves the most investigator time? | With 20 alerts a day (3.3 analyst-hours), tuned LightGBM caught 61% of fraud losses, against 29% for logistic regression: about \$2,507 recovered per analyst-hour. Ranking alerts by *probability × amount* added about 27% more recovered value per hour. | Rank queues by expected loss. Judge models on dollars recovered within capacity, because accuracy metrics only partly agree (τ = 0.73). |

## Expected impact (synthetic scenario, 0.5% fraud, fixed team of analysts)

- **Loss reduction:** Moving from a logistic-regression baseline to tuned LightGBM roughly doubles the share of fraud losses caught at the same headcount (29% → 61%). Expected-loss ranking adds more on top.
- **Workload:** No extra analysts are needed. The gains come from better ordering of the same 20 daily alerts.
- **Compute:** The tuned tree trains in minutes on a laptop and scores 1,000 transactions in a few milliseconds.

## Operational risks

1. **Tactic shifts.** Detection quality can collapse within days of fraudsters changing approach. Monitor alert precision weekly, and refit on all recent confirmed labels instead of waiting for a clean validation window.
2. **Label delay.** Recent fraud looks "legitimate" until confirmed, which hurts training and also blinds model selection. Use matured validation windows and push for faster confirmation.
3. **Default settings.** Untuned LightGBM failed outright at very low fraud rates (0.05 AUPRC at 0.1%). Tuning and regularisation aren't optional.
4. **Synthetic evidence.** These results come from simulated fraud patterns. Validate on your own data with the same protocol before changing production systems.

## Recommendation

Keep or adopt tuned gradient boosting with history and relationship features. Re-rank the alert queue by expected loss. Invest the next increment of effort in (a) weekly refits on all available labels, with drift monitoring, and (b) reducing label latency. Revisit foundation models such as TabPFN once they can be evaluated under the same protocol.
