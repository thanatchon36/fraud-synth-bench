"""Command-line interface: ``fraudlab generate|...``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from fraudlab.synth import DRIFT_RQ2, generate, preset, write


def _generate(args: argparse.Namespace) -> int:
    overrides = {"seed": args.seed}
    if args.fraud_rate is not None:
        overrides["fraud_rate"] = args.fraud_rate
    if args.noise is not None:
        overrides["noise"] = args.noise
    cfg = preset(args.preset, **overrides)
    if args.ring_share is not None:
        cfg = cfg.with_ring_share(args.ring_share)
    if args.drift == "rq2":
        cfg = replace(cfg, drift=DRIFT_RQ2)
    t0 = time.perf_counter()
    manifest = write(generate(cfg), args.out)
    summary = {
        "out": args.out,
        "content_hash": manifest["content_hash"][:16],
        "seconds": round(time.perf_counter() - t0, 2),
        **{k: manifest["stats"][k] for k in ("n_transactions", "n_fraud", "fraud_rate")},
    }
    print(json.dumps(summary))
    return 0


def _train(args: argparse.Namespace) -> int:
    from fraudlab.runner import run

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    models = args.models.split(",") if args.models else None
    print(run(args.config, root=Path(args.results_dir), seeds=seeds, models=models))
    return 0


def _probe(args: argparse.Namespace) -> int:
    from fraudlab.device import probe

    print(json.dumps(probe(args.out), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fraudlab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a synthetic dataset")
    g.add_argument("--preset", default="small", help="tiny | small | small_long | full")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--fraud-rate", type=float, default=None)
    g.add_argument("--noise", type=float, default=None)
    g.add_argument(
        "--ring-share", type=float, default=None, help="share of fraud from ring scenarios (RQ3)"
    )
    g.add_argument("--drift", choices=["none", "rq2"], default="none")
    g.set_defaults(func=_generate)

    tr = sub.add_parser("train", help="run an experiment config -> results/runs/<name>.jsonl")
    tr.add_argument("--config", required=True)
    tr.add_argument("--seeds", default=None, help="comma-separated subset, e.g. 0,1")
    tr.add_argument("--models", default=None, help="comma-separated model labels")
    tr.add_argument("--results-dir", default="results")
    tr.set_defaults(func=_train)

    pr = sub.add_parser("probe", help="hardware probe -> results/hardware.json")
    pr.add_argument("--out", default="results/hardware.json")
    pr.set_defaults(func=_probe)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
