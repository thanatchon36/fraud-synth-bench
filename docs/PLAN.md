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
