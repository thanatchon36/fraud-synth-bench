# Progress

## Phase 1: Literature review (done 2026-10-13)

**Shipped**
- `docs/literature/REVIEW.md`: 48 detailed entries across seven threads (classical, anomaly, sequence, graph, foundation-model era, operational, synthetic data), 8 foundational method references, a synthesis, and a gap-analysis table (G1–G10).
- `docs/literature/refs.bib`: 56 BibTeX entries. Every one was checked against OpenAlex (see DECISIONS D003).
- Repo skeleton: `LICENSE` (Apache-2.0), `.gitignore`, `README.md` stub, `docs/PLAN.md`, `docs/DECISIONS.md`, `docs/FUTURE_WORK.md`.

**Cut**
- Full-text reading of every paper. Summaries rely on abstracts, plus full-text checks for claims we were unsure about. Exact numbers are only quoted where confirmed.

**Open issues**
- Some entries describe metrics qualitatively ("top-k precision") where the papers use specific variants (for example, card precision@k vs. transaction precision@k).

## Phase 2: Research questions (done 2026-10-13)

**Shipped**
- `docs/RESEARCH_QUESTIONS.md`: five falsifiable RQs (old vs. new, drift, graph vs. tabular, label delay, cost and capacity), each with hypotheses, the refuting experiment, metric and numeric threshold, plus a shared statistical protocol (5 paired seeds, t-intervals, Holm correction, δ = 0.02 equivalence margin).
- DECISIONS D006–D011.

**Cut**
- The feedback loop (G10) and LLM features (G8) are out of scope for v0.1.

**Open issues**
- Thresholds such as 30% degradation (H2a) and 25% recovery (H4b) are judgement calls, not taken from the literature. They're recorded up front so they can't be changed after the results come in.
