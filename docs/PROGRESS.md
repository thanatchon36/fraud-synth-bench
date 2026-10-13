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

## Phase 3: Synthetic data generator (done 2026-10-13)

**Shipped**
- `src/fraudlab/synth/`: config and presets (`tiny`, `small`, `small_long`, `full`), the world model with legitimate behaviour and noise look-alikes, 8 registered scenario injectors (each with a docstring citing its source), and the generator with drift schedule, label delay, detection, five parquet tables, manifest and content hash.
- `fraudlab generate` CLI. Makefile targets `setup / test / test-all / lint / format / data / quick / clean`. CI workflow (lint, test, quick).
- `tests/test_synth.py`: 24 fast tests plus 1 slow test (`full` preset). Signature tests pass on seeds 0–4.
- `docs/data/DATA_CARD.md` plus the auto-generated `DATA_CARD_STATS.md`.
- Performance: `small` 0.2 s / 0.24 GB, `full` 4 s / 1.2 GB.

**Cut**
- No calibration to real aggregate statistics (FUTURE_WORK).

**Open issues**
- The difficulty check (D014) runs at the start of Phase 4, and may adjust `noise`.
- Ring scenarios have only 4–8 events each at `small` with a 0.5% fraud rate, so expect high variance in RQ3.
