# fraud-synth-bench: see README.md. All targets run inside the uv-managed env.
SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run --locked
SEEDS ?= 0 1 2 3 4

.PHONY: help setup test test-all lint format data quick probe experiments lane-a lane-b reports paper reproduce clean
CAFFEINATE := $(shell command -v caffeinate >/dev/null && echo "caffeinate -i")

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create the environment from uv.lock
	$(UV) sync --locked

test:  ## unit tests (fast)
	$(RUN) pytest -m "not slow"

test-all:  ## unit tests including the 2M-row `full` preset
	$(RUN) pytest

lint:  ## ruff lint + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## auto-format
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

data:  ## generate the `small` and `small_long` presets for every seed
	@for s in $(SEEDS); do \
		$(RUN) fraudlab generate --preset small --seed $$s --out data/small/seed$$s; \
		$(RUN) fraudlab generate --preset small_long --seed $$s --out data/small_long/seed$$s; \
	done

quick:  ## smoke run (< 5 min): tiny dataset, 2 seeds, every model family
	rm -rf results/quick
	$(RUN) fraudlab generate --preset tiny --seed 0 --out data/tiny/seed0
	$(RUN) fraudlab train --config experiments/configs/quick.yaml --results-dir results/quick

probe:  ## hardware probe -> results/hardware.json
	$(RUN) fraudlab probe

experiments: probe  ## run every RQ config: two parallel lanes (resumable; ~90 min, M4 Air)
	$(MAKE) -j2 lane-a lane-b

lane-a:  # RQ1 -> RQ4 (RQ1's FT-Transformer load is mostly on the GPU)
	@for rq in rq1 rq4; do \
		$(CAFFEINATE) $(RUN) fraudlab train --config experiments/configs/$$rq.yaml || exit 1; \
	done

lane-b:  # RQ2 -> RQ3
	@for rq in rq2 rq3; do \
		$(CAFFEINATE) $(RUN) fraudlab train --config experiments/configs/$$rq.yaml || exit 1; \
	done

reports:  ## tables, figures and results/summary.json from the run logs
	@for rq in rq1 rq2 rq3 rq4 rq5; do $(RUN) python experiments/$${rq}_report.py || exit 1; done
	$(RUN) python experiments/data_card_stats.py
	$(RUN) python experiments/build_summary.py

reproduce: ## everything from scratch: clean runs, experiments, reports, paper
	rm -rf results/runs results/timing.json results/skipped.json results/summary
	$(MAKE) experiments reports paper

paper:  ## render paper.md + EXEC_SUMMARY.md from results/summary.json, build paper.pdf + docs/paper/index.html
	$(RUN) python paper/build.py

clean:  ## remove generated data (results are kept)
	rm -rf data
