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

## Phase 4: Experiments (done 2026-10-13)

**Shipped**
- `fraudlab probe` → `results/hardware.json`.
- `fraudlab.features`: 24 tabular + 11 graph past-only features, with leakage tests (future truncation, own-label invariance, brute-force velocity).
- `fraudlab.eval`: temporal splits, metrics (AUPRC, ROC-AUC, daily P/R@k, F1 at the validation threshold, ECE, cost, latency), and the pre-registered statistics (paired t, Holm, three-way verdict).
- `fraudlab.models`: LR, RF, LightGBM (default, Optuna-tuned, + anomaly scores), Isolation Forest, autoencoder, GRU, FT-Transformer, GraphSAGE (pure PyTorch, time-respecting sampler), and TabPFN (optional, skipped without a token).
- `fraudlab.runner` + `fraudlab train`: YAML configs with three protocols (standard, drift, delay), resumable JSONL logs, timing.
- Report scripts for each RQ → `results/tables`, `results/figures`, `results/summary.json`. README headline table auto-injected.
- Headline run: all 5 RQs × 5 seeds. Logs committed in `results/runs/`.
- Verdicts: H1a not refuted, H1b refuted, H2a/H2b not refuted, H3a/H3b not refuted, H4a/H4b not refuted (H4b only through a wide CI), H5a not refuted, H5b refuted.

**Cut / changed**
- TabPFN didn't run (license token, D019). PyG replaced by plain PyTorch (D019).
- Budget measures D025, D026, D028. RQ2 retraining protocol fixed after the first run (D029). RQ4 validation confound documented (D030).

**Open issues**
- Wall time of this headline run (lanes, minutes): 104 [63, 104]. It ran with extra contention from a mid-run RQ2 rerun and an RQ4 lane change, so the fresh-clone `make reproduce` in Phase 6 is the authoritative timing.
- CI workflow still held locally (token lacks the `workflow` scope).

## Phase 5: Paper (done 2026-10-13)

**Shipped**
- `paper/paper.tmpl.md` → `paper/paper.md` → `paper/paper.pdf` (13 pages) via `make paper`. All sections from MISSION §5 are present. Every number is a placeholder rendered from `results/summary.json`, and the build fails on any missing key.
- Build pipeline: pandoc (citeproc, IEEE CSL) → Typst → PDF via the `typst` wheel. No LaTeX install needed.
- `docs/EXEC_SUMMARY.md` rendered from `docs/EXEC_SUMMARY.tmpl.md` by the same pipeline.
- Prose claims checked against the data. Five sentences were corrected before the first build (an overstated recovery, a mis-cited comparison, a speculative causal claim, and two hand-typed numbers).

**Cut**
- Table and figure cross-references by number (would need pandoc-crossref). Captions are numbered by Typst.

**Open issues**
- Pandoc renders tables with equal-width columns, so some cells wrap.
