# Plan

Plans are appended per phase, before the phase starts. Outcomes go in `PROGRESS.md`.

## Phase 1: Literature review (2026-10-13)

1. Search for fraud-detection work from 2015 to today across the seven threads listed in MISSION §1.
2. Check every citation against a bibliographic index (OpenAlex, via the API) before it goes into the review, so no reference is invented. Drop anything that can't be checked.
3. Write `docs/literature/REVIEW.md` with about 35–50 core papers. Each entry gets a summary, data, metrics, key claim and the gap it leaves open. Foundational method papers (XGBoost, GraphSAGE, …) go in a compact table instead of full entries.
4. Write `docs/literature/refs.bib` from the verified metadata.
5. End the review with a gap-analysis table (model family × data regime × evaluation condition), which feeds Phase 2.
6. Set up the repo skeleton (LICENSE, README stub, .gitignore, DECISIONS/PROGRESS/FUTURE_WORK), create a public GitHub repo, commit, and push.

There's no code yet in this phase, so `make test lint` doesn't apply. It starts with Phase 3.

## Phase 2: Research questions (2026-10-13)

1. Map gaps G1–G5 to the five default RQs from MISSION §2 and check that each is backed by the review.
2. For each RQ, write: hypothesis, the experiment that would refute it, primary metric, decision threshold, dataset preset, and models.
3. Fix the shared statistical protocol now, before any data exists, so thresholds can't be tuned after seeing results. That covers seeds, the paired test, multiple-comparison correction, and an equivalence margin.
4. Check feasibility against the compute budget (M4, 16 GB, ≤90 min suite). Resolve the conflict between the 30-day `small` preset and the 7–30-day label delay that RQ2 and RQ4 need.
