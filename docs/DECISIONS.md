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
