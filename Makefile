# fraud-synth-bench: see README.md. All targets run inside the uv-managed env.
SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run --locked
SEEDS ?= 0 1 2 3 4

.PHONY: help setup test test-all lint format data quick clean

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

quick:  ## smoke run (< 5 min): tiny dataset end to end
	$(RUN) fraudlab generate --preset tiny --seed 0 --out data/tiny/seed0

clean:  ## remove generated data (results are kept)
	rm -rf data
