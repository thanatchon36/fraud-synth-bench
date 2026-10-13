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

## Phase 3: Synthetic data generator (2026-10-13)

1. Scaffold the project: `pyproject.toml` (uv, Python 3.12, hatchling), `Makefile` (setup/test/lint/data/quick), ruff, pytest, CI workflow (lint + test).
2. Build `fraudlab.synth`:
   - `config.py`: `GeneratorConfig`, drift segments, label-delay settings, presets `tiny` (tests/quick), `small`, `small_long`, `full`.
   - `population.py`: accounts, merchants, devices, contacts, and vectorised legitimate transactions (log-normal amounts, von Mises hour-of-day, MCC mix, P2P transfers) plus legitimate look-alikes controlled by `noise` (travel, new devices, big purchases, shared household devices).
   - `scenarios/`: one class per fraud pattern behind a registry, each docstring citing its source. Patterns: card testing, account takeover, velocity abuse, mule network, merchant collusion, synthetic identity, friendly fraud, bust-out.
   - `generator.py`: fraud budget = rate × legitimate volume, per drift segment; label delay and detection; outputs `transactions`, `accounts`, `merchants`, `edges`, `labels` parquet files plus `manifest.json` (version, params, seed, content hash).
   - Independent RNG streams per component, so adding a scenario doesn't change the legitimate data.
3. Write `tests/test_synth.py` to check determinism (same seed gives the same hash, a different seed a different hash), fraud-rate tolerance, a signature statistic for each injector, the drift schedule, label-delay bounds, and schema.
4. Add the `fraudlab generate` CLI.
5. Write `docs/data/DATA_CARD.md` with statistics computed from a generated `small` dataset.
6. Measure generation time and peak memory for `small` and `full`.

## Phase 4: Experiments (2026-10-13)

1. `fraudlab probe`: device, memory, MPS op support, written to `results/hardware.json`.
2. `fraudlab.features`: past-only tabular features (amount, time, velocity windows, account and merchant age, device novelty), a separate graph feature group (accounts per device, fan-in/out, merchant breadth, known-fraud neighbours, component size and PageRank from daily snapshots), per-account sequences for the GRU, and a temporal adjacency for GraphSAGE. Includes a leakage test: perturbing future rows must not change past features.
3. `fraudlab.eval`: temporal splits, metrics (AUPRC, ROC-AUC, daily precision/recall@k, F1 at the validation threshold, ECE, cost, latency), and the stats protocol from RESEARCH_QUESTIONS (paired t, Holm, three-way verdict).
4. `fraudlab.models`: a common `fit / predict_proba / explain` interface. LR, RF, LightGBM (default and Optuna-tuned), IsolationForest, autoencoder, GRU, GraphSAGE (pure PyTorch with temporal sampling), FT-Transformer, and TabPFN (optional extra, only when the token is available).
5. The D014 difficulty check on the validation window.
6. YAML experiment configs and runner writing `results/runs/*.jsonl`. One script per RQ producing tables, figures and its slice of `results/summary.json`. Stage timing goes to `results/timing.json`.
7. Measure the suite's wall time, and trim model count or epochs if it exceeds 90 minutes, favouring clean RQ evidence.

## Phase 5: Paper (2026-10-13)

1. `paper/paper.tmpl.md`: Abstract, Introduction, Related Work, Synthetic Data Framework, Experimental Setup, Results (one subsection per RQ), Threats to Validity, Reproducibility Statement, Conclusion, References. Every number is a `{{ summary.path | fmt }}` placeholder, and tables and figures are pulled from `results/`.
2. `paper/build.py`: render the template (failing on any missing key), then pandoc (citeproc, IEEE CSL) → Typst → PDF via the `typst` wheel. No system LaTeX needed. `make paper`.
3. `docs/EXEC_SUMMARY.tmpl.md` rendered the same way: expected loss reduction, investigator workload, operational risks, recommendation.
4. Tone: neutral. Every place where a modern model didn't help is stated explicitly.

## Phase 6: Repository and publication (2026-10-13)

1. Final README (quickstart, auto-generated headline table, RQs, extension points, limitations), plus CITATION.cff.
2. `make reproduce` on a fresh clone. If it passes clean, tag v0.1.0.
3. CI workflow is ready but held locally until the gh token has the `workflow` scope (owner action).
