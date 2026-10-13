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
