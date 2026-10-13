# Research Questions

These questions come from the gap analysis in [`literature/REVIEW.md`](literature/REVIEW.md) §10. Each one is stated so it can be **refuted**. All thresholds below were fixed on 2026-10-13, **before any data was generated or any model trained**. Any later change is logged in `DECISIONS.md` with a reason.

## Shared protocol (applies to every RQ)

| Item | Choice |
|---|---|
| Seeds | 5 seeds (0–4). Each seed regenerates the dataset *and* re-seeds every model, so variance covers both data and training noise. Within a seed, all models see the identical dataset, which makes comparisons **paired**. |
| Split | Strict temporal split: train / valid / test = first 60% / next 15% / last 25% of the timeline. RQ2 and RQ4 override this as described below. No random splits anywhere. |
| Tuning | Only on the validation window. Optuna (TPE), ≤30 trials, early-stopped. The test window is touched once per (model, seed). |
| Primary metric | AUPRC (average precision) on the test window. |
| Secondary metrics | ROC-AUC, precision@k and recall@k (k = daily alert capacity), F1 at the validation-tuned threshold, ECE (15 bins), cost-weighted loss, and latency per 1k predictions. |
| Uncertainty | Mean ± std over seeds, plus a 95% CI from the t-distribution (df = 4). |
| Difference test | Paired over seeds: the per-seed difference Δ_s = metric(A) − metric(B), with a 95% t-interval for mean Δ and a paired t-test. We also report Wilcoxon signed-rank, noting that with n = 5 its two-sided p can't go below 0.0625. |
| Multiple comparisons | Holm–Bonferroni within each RQ, over that RQ's pre-registered comparisons. |
| Practical-significance margin | **δ = 0.02 AUPRC (absolute).** "A beats B" requires Holm-adjusted p < 0.05 **and** mean Δ ≥ δ. "A and B are equivalent" requires the 95% CI of Δ to lie inside [−δ, +δ]. Otherwise the result is **inconclusive**, and we report it that way. |

**Dataset presets** (exact parameters live in the generator, Phase 3):
- `small`: about 200k transactions, 20k accounts, 30 days. Used for RQ1, RQ3 and RQ5.
- `small_long`: about 200k transactions, 8k accounts, 90 days. Used for RQ2 and RQ4. The row budget, and so the compute, matches `small`, but the timeline is long enough for 7–30-day label delays and a mid-stream drift. See DECISIONS D007.
- `full`: about 2M transactions. Documented and tested but not used in the paper.

---

## RQ1: Old vs. new under extreme imbalance  *(gap G1)*

**Question.** At fraud rates of 0.1–1%, do modern tabular models (TabPFN v2, FT-Transformer) beat a *tuned* gradient-boosted tree on AUPRC, or does any advantage disappear once the GBDT is tuned?

- **H1a (main):** Tuned LightGBM is **not beaten** by TabPFN or FT-Transformer at any fraud rate, meaning no modern model meets the "beats" criterion against tuned LightGBM.
- **H1b:** Tuning matters. Tuned LightGBM beats default-parameter LightGBM by ≥ δ at a fraud rate of 0.1%.
- **Experiment.** Preset `small`, fraud rates {0.1%, 0.5%, 1.0%} (target share of transactions), 5 seeds. Models: logistic regression, random forest, LightGBM (default), LightGBM (tuned), TabPFN v2, FT-Transformer, plus the unsupervised IsolationForest and autoencoder for reference. All models get the same engineered tabular features (amount, time, per-account velocity and aggregation windows).
- **TabPFN constraint.** TabPFN is trained on a ≤10k-row **stratified temporal subsample**: all training-window frauds (up to 50% of the cap), and legitimate transactions sampled uniformly over time. This is logged as a model-specific constraint and treated as a reportable finding ("needs subsampling; GBDT doesn't").
- **What refutes H1a.** Either modern model meets the "beats" criterion against tuned LightGBM at any fraud rate.
- **What refutes H1b.** Tuned vs. default LightGBM is equivalent or inconclusive at 0.1%.
- **Metric.** Test AUPRC (primary). Calibration (ECE) and latency per 1k are reported as secondary.

## RQ2: Robustness to fraudster adaptation (concept drift)  *(gap G2)*

**Question.** When fraudsters switch tactics mid-stream, which model family degrades least, and how much does periodic retraining recover?

- **Setup.** Preset `small_long` (90 days). Drift schedule: up to **t = 60% (day 54)**, fraud comes from scenarios set A = {card testing, account takeover, velocity abuse}. After that point, it comes from set B = {bust-out, synthetic identity, mule network}. The overall fraud rate stays at 0.5%. Static models train on days 0–40, validate on days 40–50, and are tested in weekly windows over days 50–90.
- **H2a:** All supervised families lose ≥ 30% of their pre-drift AUPRC in the post-drift windows. The unsupervised detectors (IsolationForest, autoencoder) lose < 30%, since they don't depend on old fraud labels.
- **H2b:** Weekly retraining on all labels available at retrain time (with label delay off in RQ2, so drift is isolated from delay) recovers ≥ 50% of the lost AUPRC within 14 days of the drift, for LightGBM.
- **Measures.** Relative degradation D = 1 − AUPRC_post / AUPRC_pre, where pre = days 50–54 and post = days 54–90. Recovery R = (AUPRC_retrain − AUPRC_static) / (AUPRC_pre − AUPRC_static), measured over days 61–68.
- **What refutes H2a.** Any supervised family with an upper 95% CI of D below 0.30, or any unsupervised family with a lower 95% CI of D above 0.30.
- **What refutes H2b.** An upper 95% CI of R for LightGBM below 0.50.
- **Families compared.** Logistic regression, LightGBM (tuned), GRU (sequence), IsolationForest, autoencoder.

## RQ3: Graph models vs. tabular models with graph features  *(gap G3)*

**Question.** Does collusive or ring fraud (shared devices, mule accounts, merchant collusion) give a GNN a measurable edge over a tabular model fed hand-crafted graph features?

- **Setup.** Preset `small`, fraud rate 0.5%. We vary the **ring share**, the fraction of fraud from graph-structured scenarios (mule network, merchant collusion, shared-device synthetic identities): {0%, 25%, 50%, 75%}. The graph is built **only from edges whose timestamp is ≤ the prediction time**, so no future edges leak in (following `lu2022bright`).
- **Models.** LightGBM without graph features; LightGBM plus graph features (account degree, shared-device count, shared-device neighbours' fraud-label rate from labelled training data only, PageRank, connected-component size); GraphSAGE (mini-batch neighbour sampling, CPU).
- **H3a:** At every ring share, LightGBM + graph features is **equivalent to or better than** GraphSAGE.
- **H3b:** Graph information matters for ring fraud. LightGBM + graph features beats LightGBM without graph features at ring shares ≥ 50%.
- **What refutes H3a.** GraphSAGE beats LightGBM + graph features at any ring share.
- **What refutes H3b.** Equivalence or an inconclusive result at both 50% and 75%.
- **Metric.** Test AUPRC on all transactions, plus recall@k on ring-scenario frauds only.

## RQ4: Label delay  *(gap G4)*

**Question.** How much does realistic label latency (fraud confirmed 7–30 days after the event) hurt each model family, and does unsupervised or semi-supervised pretraining reduce the damage?

- **Setup.** Preset `small_long`, fraud rate 0.5%, no drift. The model is trained at day T = 60 and tested on days 60–90. Two label regimes at training time:
  - *Oracle*: every training label is known at T.
  - *Delayed*: a fraud label is visible only if `label_available_at ≤ T`, where the delay ~ uniform(7, 30) days. Unconfirmed transactions are treated as legitimate, as production systems do.
- **Models.** LightGBM (tuned), logistic regression, GRU. A **semi-supervised variant**, LightGBM + unsupervised scores, gets IsolationForest and autoencoder anomaly scores as features. The detectors are fit on *all* transactions up to T, labelled or not.
- **H4a:** Label delay reduces the AUPRC of every supervised family by ≥ 10% relative to the oracle.
- **H4b:** The semi-supervised variant recovers ≥ 25% of the AUPRC lost by plain LightGBM: recovery = (AUPRC_semi,delayed − AUPRC_lgbm,delayed) / (AUPRC_lgbm,oracle − AUPRC_lgbm,delayed).
- **What refutes H4a.** Any supervised family where the upper 95% CI of the relative loss is below 10%.
- **What refutes H4b.** An upper 95% CI of recovery below 25%.

## RQ5: Cost and investigator capacity  *(gap G5)*

**Question.** Under a cost matrix where a missed fraud costs ≫ a false alert, and investigators can review only k alerts per day, which model recovers the most loss per investigator-hour? Does ranking models by AUPRC give the same answer?

- **Cost model.** A missed fraud costs its transaction amount plus a $25 chargeback fee. An alert costs 10 investigator-minutes. Capacity is k = 20 alerts per day. In `small` at a 0.5% fraud rate there are about 6.7k transactions and about 33 frauds per day, so capacity is below the daily fraud count, which forces a choice.
- **Primary metric.** Recovered loss per investigator-hour = (amount of fraud in alerted transactions + its fees) / (alerts reviewed × 10/60 hours). Capacity is fixed, so this is a value-weighted precision@k.
- **Ranking policies.** (i) rank by p(fraud), and (ii) rank by expected loss p(fraud) × amount. We evaluate both for every RQ1 model, using the RQ1 runs at a 0.5% fraud rate.
- **H5a:** Ranking by expected loss gives ≥ 10% more recovered loss per investigator-hour than ranking by p(fraud), for the best model.
- **H5b:** Model ordering by cost metric matches ordering by AUPRC. The Kendall τ between the two rankings (mean over seeds) is ≥ 0.8.
- **What refutes H5a.** An upper 95% CI of the relative gain below 10%.
- **What refutes H5b.** Mean τ < 0.8. That would be a substantive finding: AUPRC would then be a poor proxy for operational value.

---

## Out of scope for v0.1 (see `FUTURE_WORK.md`)
- Investigator feedback loops and selection bias (G10).
- LLM-derived features (G8). Ollama is only attempted if the core suite fits the time budget with margin.
