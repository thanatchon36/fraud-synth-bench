# Decision Log

Each entry records what was decided, why, and which alternatives were rejected. Entries are in chronological order.

---

### D001: Repository name `fraud-synth-bench` (2026-10-13, Phase 1)
- **What:** The public GitHub repo is `thanatchon36/fraud-synth-bench`. The Python package is `fraudlab`.
- **Why:** MISSION §6 names this layout. The name says what the project is (a benchmark on synthetic fraud data) rather than where it lives locally (`fraud-analytics`).
- **Rejected:** `fraud-analytics`, which is too generic and collides with many existing repos.

### D002: Commit identity uses the GitHub noreply address (2026-10-13, Phase 1)
- **What:** Commits are authored as `Thanatchon Chongmankhong <10392423+thanatchon36@users.noreply.github.com>`, set in the repo-local git config.
- **Why:** No global git identity was configured. The noreply address keeps the owner's personal email out of a public history, in line with MISSION §0 (no PII).
- **Rejected:** Using a personal email address.

### D003: Every citation is checked against a bibliographic index (2026-10-13, Phase 1)
- **What:** Each paper in `docs/literature/` was checked against OpenAlex (title, authors, year, venue, DOI/arXiv ID) before inclusion. Detailed claims we weren't sure about (for example, the Elliptic dataset statistics and SAINT's use of the ULB credit card dataset) were checked against the paper's full text.
- **Why:** Invented references are the most common failure of LLM-assisted literature reviews, and honest science is a hard constraint.
- **Rejected:** Writing from memory alone. One candidate paper on delayed labels in non-stationary streams couldn't be confirmed in OpenAlex, so it was dropped and replaced with the Gomes et al. (2022) survey.
- **Note:** The arXiv and Semantic Scholar APIs returned rate-limit errors (HTTP 429 or 503) during the review, so OpenAlex was the only index used.

### D004: About 50 core papers plus a separate table of foundational methods (2026-10-13, Phase 1)
- **What:** The review has 48 detailed entries across 7 threads, somewhat above the ~20–40 target, because the operational thread (drift, delay, cost) needed its own grounding. Eight foundational method papers (XGBoost, LightGBM, CatBoost, Isolation Forest, OC-SVM, VAE, GraphSAGE, GAT) are listed in a compact table without full summaries.
- **Why:** MISSION asks for 20–40 papers with gap analysis. Method papers have no fraud-specific "gap", so full entries would be padding.
- **Rejected:** A longer, shallower list.

### D005: Pre-2015 papers allowed as background (2026-10-13, Phase 1)
- **What:** A few influential pre-2015 papers are included (Dal Pozzolo 2014, Gama 2014, He & Garcia 2009, Elkan 2001, Liu 2008, Schölkopf 2001).
- **Why:** They define the operational framing (alert precision, drift, cost) that later fraud work relies on. Leaving them out would make the gap analysis look less grounded than it is.
- **Rejected:** A strict 2015 cutoff.

### D006: Thresholds pre-registered with a practical margin δ = 0.02 AUPRC (2026-10-13, Phase 2)
- **What:** Every RQ has a numeric refutation criterion, fixed before data exists. "Beats" requires Holm-adjusted p < 0.05 **and** mean Δ ≥ 0.02. "Equivalent" requires the 95% CI of Δ to lie within ±0.02. Anything else is reported as inconclusive.
- **Why:** With 5 seeds, p-values alone are fragile, and tiny AUPRC gaps have no operational meaning. A three-way verdict (beats / equivalent / inconclusive) stops us from reading noise as a finding.
- **Rejected:** p-values only. Wilcoxon only (with n = 5, the two-sided minimum p is 0.0625, so it can never reach significance). Bootstrapping over test rows only (that ignores seed variance).

### D007: `small_long` preset for the time-dependent RQs (2026-10-13, Phase 2)
- **What:** RQ2 and RQ4 use a 90-day, 8k-account preset with about the same number of transactions (about 200k) as `small`.
- **Why:** In a 30-day timeline, a 7–30-day label delay would hide almost every training label, and a drift at day 18 would leave too little post-drift data. Keeping the row count fixed keeps compute in line with `small`.
- **Rejected:** `full` for RQ2/RQ4 (10× the compute and over the 90-minute budget). Shrinking the delay to fit 30 days (that departs from the MISSION's 7–30-day realism).

### D008: TabPFN gets a ≤10k-row stratified temporal subsample (2026-10-13, Phase 2)
- **What:** TabPFN's training set keeps all training-window frauds (capped at 50% of 10k) plus legitimate transactions sampled uniformly over time.
- **Why:** TabPFN v2 is designed for ≤10k samples and runs CPU-only on this machine. A uniform subsample at 0.1% fraud would keep only about 10 positives. Stratifying gives it a fair chance, and the need to subsample is itself a reportable finding for RQ1.
- **Rejected:** Uniform subsampling (unfair to TabPFN). Chunked ensembling over the full data (non-standard, and slow on CPU).

### D009: Unconfirmed labels are treated as legitimate (RQ4) (2026-10-13, Phase 2)
- **What:** Under label delay, a fraud whose label hasn't arrived by training time is labelled 0.
- **Why:** This matches how production fraud systems train ("no chargeback yet" counts as genuine), and it's the mechanism that makes delay harmful.
- **Rejected:** Dropping recent unlabelled rows. That's a valid alternative mitigation, but it would confound H4b.

### D010: Drift and label delay are tested separately (2026-10-13, Phase 2)
- **What:** RQ2 runs without label delay, and RQ4 runs without drift.
- **Why:** Interacting the two would make it impossible to attribute any degradation to one cause. The interaction is listed in FUTURE_WORK.
- **Rejected:** One combined realistic scenario as the main experiment.

### D011: RQ5 capacity of k = 20 alerts/day (2026-10-13, Phase 2)
- **What:** Investigator capacity is 20 alerts per day, with 10 minutes per alert, a $25 chargeback fee, and missed-fraud loss equal to the transaction amount.
- **Why:** Capacity has to be below the daily fraud count (about 33/day in `small` at 0.5%), otherwise ranking quality barely matters. A first draft used k = 50, which exceeds the fraud count, so it was corrected before any data existed.
- **Rejected:** Capacity as a percentage of volume. A fixed headcount is how fraud teams actually plan.

### D012: numpy for simulation, Polars for tables and I/O (2026-10-13, Phase 3)
- **What:** The generator simulates with vectorised numpy arrays (`TxBatch`) and emits Polars DataFrames with `Enum` categoricals, written as zstd parquet.
- **Why:** `full` (2M rows) generates in about 4 s with a 1.2 GB peak RSS, far under the 10 GB cap. Polars `Enum` keeps categoricals compact and stable.
- **Rejected:** pandas (higher memory for strings). A per-agent Python event loop like PaySim (orders of magnitude slower).

### D013: Exact fraud budgets by trimming a time-ordered prefix (2026-10-13, Phase 3)
- **What:** Injectors generate whole events until the budget is met, then drop the time-ordered tail of the last event.
- **Why:** An exact realised fraud rate is needed for RQ1's rate sweep (0.1% has only about 200 fraud transactions in `small`). Time-ordering means a trimmed mule ring keeps inflows before outflows, as if the event was interrupted.
- **Rejected:** Accepting overshoot (up to ±50% per scenario at 0.1%). Trimming in insertion order (that would cut all of a ring's outflows).

### D014: Pre-registered difficulty check before any RQ experiment (2026-10-13, Phase 3)
- **What:** Before running RQ experiments, train **default LightGBM** on `small` seed 0 and score the **validation** window only. If its AUPRC is outside **[0.30, 0.85]**, the generator is too easy or too hard to tell models apart. In that case only the `noise` knob may be adjusted (and logged here), and the check is repeated. The test window isn't looked at.
- **Why:** A saturated benchmark (AUPRC ≈ 1) would make every RQ trivially "equivalent". A hopeless one would make every result noise. Fixing the target range and the single permitted knob in advance stops the data being tuned toward a favoured result.
- **Rejected:** Free-form tuning of generator parameters after seeing model comparisons.

### D015: Card-present payments have no customer device (2026-10-13, Phase 3)
- **What:** `device_id` is null for `channel = pos`.
- **Why:** At a physical terminal, the issuer doesn't observe a customer device. This makes device-based signals (ATO, shared devices) available only for online traffic and transfers, as in practice.

### D016: Card testers choose merchants uniformly (2026-10-13, Phase 3)
- **What:** The card-testing injector samples online-only merchants uniformly, not by popularity.
- **Why:** Found by the multi-seed signature check. With popularity weighting, tiny datasets concentrated a burst on 1–2 merchants, which contradicts the pattern (testers spread over many low-profile merchants).

### D017: The `accounts.account_role` ground truth is shipped in the raw tables (2026-10-13, Phase 3)
- **What:** `account_role` (customer / mule / collector / synthetic) is in `accounts.parquet`, and is documented in the DATA_CARD as ground truth that must not be used as a feature.
- **Why:** Analysis needs it (for example, recall on mule accounts). The feature builders in Phase 4 use an explicit allow-list of columns, so it can't leak by accident.

### D014 outcome: noise lowered from 0.5 to 0.4 (2026-10-13, Phase 4)
- **What:** The difficulty check (default LightGBM, `small`, seed 0, oracle labels, tabular features, validation window days 18–22) gave a validation AUPRC of **0.297 at noise 0.5**, just below the pre-registered lower bound of 0.30. Following the rule, `noise` was lowered by one step to **0.4**, which gave **0.551**. That is inside [0.30, 0.85], so 0.4 is now the default for every preset. The test window wasn't evaluated during this check.
- **Observation, recorded for later:** a 0.1 change in noise moved validation AUPRC by 0.25. The validation window holds only 106 fraud transactions from a few dozen events, and untuned LightGBM is unstable at 0.5% prevalence. Expect high seed variance, which is why every RQ uses 5 paired seeds.

### D018: Legitimate P2P transfers use round amounts (2026-10-13, Phase 4, before any experiment)
- **What:** 50% of legitimate transfers are rounded to 10 and 20% to 50.
- **Why:** The first feature build showed `is_round` was 0.0 for all legitimate traffic, which would make merchant collusion's round amounts a free giveaway. People really do send round sums.

### D019: Optional and unavailable dependencies isolated (2026-10-13, Phase 4)
- **TabPFN:** `tabpfn` 9.2 (Apache-2.0 code) downloads its weights only after a one-time license acceptance tied to a Prior Labs account (`TABPFN_TOKEN`). That's a credential only the repository owner can supply. TabPFN is therefore an optional extra (`uv sync --extra tabpfn`), and the runner skips it and records the skip in `results/skipped.json` when the token is absent. **RQ1's H1a is consequently tested against FT-Transformer only in the default reproduction.** The paper says so explicitly.
- **PyTorch Geometric:** PyG's neighbour samplers need `pyg-lib` or `torch-sparse`, which have no PyPI wheels for macOS arm64 (checked with `uv pip install --dry-run`). GraphSAGE (mean aggregator, uniform fixed-fan-out sampling, as in Hamilton et al. 2017) is implemented in about 150 lines of plain PyTorch, with a **time-respecting** sampler that PyG would also need `pyg-lib` for.
- **Rejected:** Building `torch-sparse` from source (fragile, not reproducible on a fresh Mac). Downloading TabPFN weights by other means (that would bypass the license).

### D020: Label-derived features always use realistic label availability (2026-10-13, Phase 4)
- **What:** Features such as `acc_known_fraud_cnt` and `dev_known_fraud_cnt` count only fraud whose `label_available_at` is before min(T, deployment time), whatever the target label mode.
- **Why:** With instant (oracle) labels, training rows would see fraud labels from earlier in the *same* event, which test rows never can. That's a train/test feature shift that inflates training fit and hurts generalisation.

### D021: Cost model of $40/h investigators and 10 min per alert (2026-10-13, Phase 4)
- **What:** Alert cost = $40/h × 10 min = $6.67. Missed fraud = amount + $25 fee. The cost metrics use these values. RQ5's primary metric (recovered $ per investigator-hour) doesn't depend on the hourly rate.
- **Why:** Round, plausible operating figures. They aren't drawn from any real institution.

### D022: Per-model compute device chosen by measurement (2026-10-13, Phase 4)
- **What:** Training time on `small`, seed 0, 2 epochs, MPS vs. CPU: autoencoder 2.2 s vs **1.0 s**; GRU **3.4 s** vs 5.2 s; FT-Transformer **31.6 s** vs 127.5 s. The autoencoder runs on CPU, while GRU and FT-Transformer use MPS (`torch_device()` default). GraphSAGE always runs on CPU (MISSION §0). Test AUPRC agreed across devices to two decimals (GRU 0.272 vs 0.273).
- **Why:** MISSION asks for MPS when available, but at these model sizes kernel-launch overhead makes MPS slower for the tiny autoencoder. `FRAUDLAB_DEVICE` overrides the choice everywhere.

### D023: torch runs CPU kernels single-threaded (2026-10-13, Phase 4)
- **What:** `torch.set_num_threads(1)` at import, and `fraudlab/__init__.py` imports LightGBM before torch.
- **Why:** Three OpenMP runtimes end up in one process: LightGBM's (Homebrew libomp), scikit-learn's and PyTorch's. Reproduced failures: importing torch before LightGBM segfaults in `lgb.train`, and multithreaded torch CPU ops after `sklearn.ensemble` is imported segfault. With single-threaded torch, every combination ran cleanly. LightGBM and scikit-learn keep 8 threads, and the heavier neural models use MPS, which doesn't use OpenMP.
- **Consequence:** A fresh Mac needs `brew install libomp` for LightGBM (README quickstart). Scripts must import `fraudlab` before `torch`.

### D024: The label cutoff is the deployment time (test start) (2026-10-13, Phase 4)
- **What:** Training **and validation** labels are those known before the test window starts. Label-derived features for test rows use only labels known before deployment.
- **Why:** The first draft cut labels at the start of the validation window, which made every validation label 0. Tuning and early stopping then silently optimised noise (found in the first benchmark: tuned LightGBM scored below default). This describes a model trained and tuned on data up to deployment, then frozen.

### D025: Neural models train on per-epoch negative undersampling (2026-10-13, Phase 4)
- **What:** GRU, FT-Transformer and GraphSAGE train each epoch on all positives plus a fresh 10% sample of negatives, for up to 20 epochs with patience 3 on validation AUPRC. Predictions are corrected with the closed form of Dal Pozzolo et al. (2015), p = βp_s / (βp_s − p_s + 1). This leaves rankings unchanged and restores calibration.
- **Why (validation evidence):** In the first suite launch (stopped after two RQ1 cells), FT-Transformer's **validation** AUPRC was still creeping up when it hit its 6-epoch cap (0.001 → 0.007), and the GRU was still improving at its 8-epoch cap. Both were under-trained rather than converged, and full epochs cost about 15 s each for FT-Transformer, which can't fit the time budget. Undersampling makes an epoch about 10× cheaper, so the same budget buys convergence.
- **Disclosure:** Two FT-Transformer **test** AUPRC values (0.003 and 0.045 at 0.1% fraud) were visible in that aborted run's log before this change. The change was motivated by the validation curves and the time budget above, applies identically to all three neural models, and was made without any further look at test results. All reported results come from a clean rerun.
- **Rejected:** Class-weighted BCE (no speed-up). Tuning neural hyperparameters with Optuna (budget).

### D026: LightGBM search space capped for the time budget (2026-10-13, Phase 4)
- **What:** Learning rate in [0.02, 0.3] (was [0.01, 0.2]), at most 1,000 boosting rounds (was 2,000), early stopping after 30 rounds (was 50). Still 30 TPE trials for RQ1. RQ2 retrains use 10 trials.
- **Why:** In the aborted run, one tuned fit took 218 s because low learning rates ran up to about 1,350 rounds. The whole suite projected to over 3 hours against a 90-minute budget. Narrowing the range removes the slowest corner of the space. Validation AUPRC on those two cells came from moderate learning rates (0.074 and 0.016).

### D027: Observation: default LightGBM degenerates at 0.1% fraud (2026-10-13, Phase 4)
- **What:** Default LightGBM on `small` at 0.1% fraud (114 training positives) produced predictions saturated at exactly 0 and 1 (351 distinct values), training AUPRC 0.14, and test ranking no better than random.
- **Why it matters:** With `lambda_l2 = 0` and a tiny minimum hessian per leaf, leaf values explode when positives are this rare. This is genuine default behaviour, not a pipeline bug. It's what H1b ("tuning matters") is about, and it's reported as such rather than "fixed".

### D028: Time-budget measures: two parallel lanes, 15 trials for RQ3/RQ4, fixed thread counts (2026-10-13, Phase 4)
- **What:** `make experiments` runs RQ1 → RQ4 in one process and RQ2 → RQ3 in a second. (The first launch used RQ1 | RQ2 → RQ3 → RQ4. Lane B ran long, so RQ4 was started in lane A once RQ1 finished, and the Makefile was updated to match.) LightGBM, Random Forest and Isolation Forest are fixed at 5 threads in both serial and parallel runs, so results don't depend on how the suite is scheduled. In RQ3 and RQ4, every LightGBM arm (with and without graph features, plus the semi-supervised variant) is tuned with 15 Optuna trials. RQ1 keeps the full 30, because H1 is specifically about tuned GBDT.
- **Why:** A timed RQ1 cell after D025/D026 took about 150 s (tuned LightGBM 46 s, FT-Transformer 85 s), projecting about 117 minutes serially. The comparisons inside RQ3 and RQ4 are between LightGBM arms with *equal* budgets, so halving both keeps them fair. MISSION's "cleaner evidence over more models" rule kept every RQ instead of dropping one.
- **Rejected:** Fewer seeds (weakens every test). Dropping FT-Transformer (it's RQ1's main modern arm while TabPFN is unavailable).

### D029: RQ2 retraining fixed to match the pre-registered H2b; holdout variant kept and reported (2026-10-13, Phase 4)
- **What happened:** The first RQ2 run implemented "weekly retraining" as a full re-tune at each retrain day *r*: train on [0, r − 10), validate on [r − 10, r). The report showed recovery R = −0.07 (95% CI [−0.14, −0.00]). Investigating, the cause was the protocol, not the models. The drift is on day 54, so the retrains serving the recovery window (days 61–68, retrained on days 57 and 64) trained on data ending at days 47 and 54. **They never saw a single post-drift fraud**, and recovery was ≈0 by construction.
- **Why that's a bug:** H2b, as pre-registered, is about retraining "on all labels available at retrain time". The holdout implementation didn't do that.
- **Fix:** A `refit` policy that, at each retrain day, refits LightGBM on **all** rows before *r*, with the hyperparameters and round count chosen at deployment (`LightGBMTuned.refit`). H2b is evaluated on this policy.
- **Kept:** The holdout `retrain` policy is still run and reported, because it shows something practically useful: a validation holdout delays adaptation by its own length.
- **Disclosure:** This fix was made *after* seeing the RQ2 recovery estimate for the holdout policy. The refit policy is the one the pre-registration text describes. The decision threshold (R ≥ 0.5) is unchanged.

### D030: Verdicts are "refuted / not refuted", plus the RQ4 validation confound (2026-10-13, Phase 4)
- **Verdict language:** The pre-registration defines only *refutation* criteria. Reports therefore say "refuted" or "not refuted", never "supported". Where a hypothesis survives only because its interval is wide (for example, H4b, whose point estimate points the other way), the paper says so explicitly.
- **RQ4 confound (found after the run, from the generator, without looking at test results):** In the delayed-label condition, the validation window (days 50–60) has 0 visible frauds at deployment in 4 of 5 seeds (8 in the remaining one), against 52–148 with oracle labels. Delayed-label models are therefore tuned and early-stopped with essentially no validation signal. RQ4's "delayed" effect combines fewer training labels with blind model selection. The paper reports it as such, and FUTURE_WORK lists delay-aware validation. The counts are computed by `experiments/rq4_report.py` (`validation_positives`).
