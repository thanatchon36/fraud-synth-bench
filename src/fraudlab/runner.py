"""Experiment runner: YAML config → ``results/runs/<name>.jsonl``.

A config is dataset params × conditions × models × evaluation protocol × seeds.
Every run (one model, one seed, one condition) appends a JSON line, and runs that
already exist are skipped, so an interrupted experiment resumes where it stopped.

Protocols:

* ``standard``: temporal 60/15/25 split, deployment at the test start (RQ1, RQ3, RQ5).
* ``drift``: static vs. weekly-retrained models, scored per test window (RQ2).
* ``delay``: train/valid/test cut by day, labels oracle or delayed (RQ4).
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import yaml

from fraudlab.eval import TemporalSplit, best_f1_threshold, daily_top_k, evaluate
from fraudlab.features import LabelView
from fraudlab.models import MODELS, ModelData, build_model_data, tabpfn_available
from fraudlab.synth import DRIFT_RQ2, generate, preset

RESULTS = Path("results")


# ----------------------------------------------------------------------------- config


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("protocol", "standard")
    cfg.setdefault("seeds", [0, 1, 2, 3, 4])
    cfg.setdefault("conditions", {})
    cfg.setdefault("label_mode", "oracle")
    return cfg


def _conditions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = cfg["conditions"]
    keys = sorted(grid)
    return [
        dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))
    ]


def _dataset(cfg: dict[str, Any], cond: dict[str, Any], seed: int):
    ds_cfg = dict(cfg["dataset"])
    name = ds_cfg.pop("preset")
    drift = ds_cfg.pop("drift", None)
    overrides = {
        **ds_cfg,
        **{k: v for k, v in cond.items() if k not in ("ring_share", "label_mode")},
    }
    gcfg = preset(name, seed=seed, **overrides)
    if "ring_share" in cond:
        gcfg = gcfg.with_ring_share(cond["ring_share"])
    if drift == "rq2":
        gcfg = replace(gcfg, drift=DRIFT_RQ2)
    return generate(gcfg)


def _key(cond: dict[str, Any], seed: int, label: str) -> str:
    return json.dumps({"c": cond, "s": seed, "m": label}, sort_keys=True)


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out.add(_key(r["condition"], r["seed"], r["model"]))
    return out


# ----------------------------------------------------------------------------- helpers


def _rows(md: ModelData, start: datetime, a: datetime, b: datetime) -> np.ndarray:
    lo = (a - start).total_seconds()
    hi = (b - start).total_seconds()
    return np.flatnonzero((md.t >= lo) & (md.t < hi))


def _per_scenario_recall(md: ModelData, idx: np.ndarray, score: np.ndarray, k: int) -> dict:
    """Share of each scenario's test frauds that land in the daily top-``k`` alerts."""
    alert = daily_top_k(score, md.day[idx], k)
    scen = md.frame["scenario"].cast(pl.String).to_numpy()[idx]
    fraud = md.y_true[idx] == 1
    return {s: float(alert[(scen == s) & fraud].mean()) for s in np.unique(scen[fraud])}


def _model_spec(spec: dict[str, Any] | str) -> dict[str, Any]:
    spec = {"name": spec} if isinstance(spec, str) else dict(spec)
    spec.setdefault("features", "tabular")
    spec.setdefault(
        "label",
        spec["name"]
        if spec["features"] == "tabular"
        else f"{spec['name']}+{spec['features'].split('+')[-1]}",
    )
    spec.setdefault("params", {})
    return spec


def _fit_eval(
    spec: dict,
    md: ModelData,
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    cls = MODELS[spec["name"]]
    data = md.with_features(spec_features(spec))
    model = cls(seed=seed, **spec["params"])
    t0 = time.perf_counter()
    model.fit(data, train, valid)
    fit_s = time.perf_counter() - t0
    p_valid = model.predict_proba(data, valid)
    t1 = time.perf_counter()
    p_test = model.predict_proba(data, test)
    pred_s = time.perf_counter() - t1
    thr = best_f1_threshold(data.y_train[valid], p_valid) if data.y_train[valid].sum() else None
    metrics = evaluate(
        md.y_true[test],
        p_test,
        md.amount[test],
        md.day[test],
        threshold=thr,
        probabilistic=model.probabilistic,
    )
    metrics["valid_auprc"] = _auprc_or_nan(data.y_train[valid], p_valid)
    metrics["latency_ms_per_1k"] = 1000 * pred_s / len(test) * 1000
    return {
        "model": spec["label"],
        "model_class": spec["name"],
        "family": model.family,
        "features": spec["features"],
        "metrics": metrics,
        "recall_at_k_by_scenario": _per_scenario_recall(md, test, p_test, 20),
        "info": _jsonable(model.info),
        "fit_seconds": fit_s,
        "predict_seconds": pred_s,
        "explain": _jsonable(model.explain(data, valid)) if spec.get("explain") else None,
        "_scores": p_test,
    }


def spec_features(spec: dict) -> tuple[str, ...]:
    from fraudlab.models import FEATURE_SETS

    return FEATURE_SETS[spec["features"]]


def _auprc_or_nan(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if y.sum() else float("nan")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float | int | str | bool) or x is None:
        return x
    return str(x)


def _append(path: Path, row: dict[str, Any]) -> None:
    row = {k: v for k, v in row.items() if not k.startswith("_")}
    with path.open("a") as f:
        f.write(json.dumps(_jsonable(row)) + "\n")


def _rss_gb() -> float:
    return round(psutil.Process().memory_info().rss / 2**30, 2)


# ----------------------------------------------------------------------------- protocols


def run(
    config_path: str | Path,
    root: Path = RESULTS,
    seeds: list[int] | None = None,
    models: list[str] | None = None,
) -> Path:
    cfg = load_config(config_path)
    out_dir = root / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cfg['name']}.jsonl"
    done = _done(out)
    specs = [_model_spec(s) for s in cfg["models"]]
    if models:
        specs = [s for s in specs if s["label"] in models or s["name"] in models]
    ok, why = tabpfn_available()
    if not ok and any(s["name"] == "tabpfn" for s in specs):
        print(f"[runner] skipping tabpfn: {why}")
        _note_skip(root, cfg["name"], "tabpfn", why)
        specs = [s for s in specs if s["name"] != "tabpfn"]
    protocol = {"standard": _run_standard, "drift": _run_drift, "delay": _run_delay}[
        cfg["protocol"]
    ]
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    for cond in _conditions(cfg) or [{}]:
        for seed in seeds if seeds is not None else cfg["seeds"]:
            todo = [s for s in specs if _key(cond, seed, s["label"]) not in done]
            if not todo:
                continue
            protocol(cfg, cond, seed, todo, out, timing)
            gc.collect()
    timing["total"] = time.perf_counter() - t0
    _write_timing(root, cfg["name"], timing)
    return out


def _note_skip(root: Path, name: str, model: str, why: str) -> None:
    path = root / "skipped.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[f"{name}/{model}"] = why
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _write_timing(root: Path, name: str, timing: dict[str, float]) -> None:
    path = root / "timing.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    prev = data.get(name, {})
    # accumulate across resumed invocations
    data[name] = {k: round(prev.get(k, 0.0) + v, 2) for k, v in timing.items()} | {
        k: v for k, v in prev.items() if k not in timing
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _tick(timing: dict[str, float], key: str, seconds: float) -> None:
    timing[key] = timing.get(key, 0.0) + seconds


def _base_row(cfg, cond, seed, ds) -> dict[str, Any]:
    return {
        "experiment": cfg["name"],
        "condition": cond,
        "seed": seed,
        "preset": ds.config.preset,
        "dataset_hash": ds.content_hash()[:16],
        "fraud_rate": ds.config.fraud_rate,
    }


def _run_standard(cfg, cond, seed, specs, out, timing) -> None:
    t = time.perf_counter()
    ds = _dataset(cfg, cond, seed)
    _tick(timing, "generate", time.perf_counter() - t)
    start = datetime.fromisoformat(ds.config.start_date)
    split = TemporalSplit.from_fractions(start, ds.config.n_days)
    t = time.perf_counter()
    md = build_model_data(
        ds,
        LabelView(split.label_cutoff, cond.get("label_mode", cfg["label_mode"])),
        "tabular+graph",
    )
    _tick(timing, "features", time.perf_counter() - t)
    train = _rows(md, start, split.start, split.valid_start)
    valid = _rows(md, start, split.valid_start, split.test_start)
    test = _rows(md, start, split.test_start, split.end)
    for spec in specs:
        r = _fit_eval(spec, md, train, valid, test, seed)
        _tick(timing, f"model/{spec['label']}", r["fit_seconds"] + r["predict_seconds"])
        _append(
            out,
            _base_row(cfg, cond, seed, ds)
            | r
            | {
                "n_train": len(train),
                "n_valid": len(valid),
                "n_test": len(test),
                "n_test_fraud": int(md.y_true[test].sum()),
                "rss_gb": _rss_gb(),
            },
        )
        print(
            f"[{cfg['name']}] {cond} seed={seed} {spec['label']:22s} "
            f"auprc={r['metrics']['auprc']:.3f} fit={r['fit_seconds']:.1f}s",
            flush=True,
        )


def _run_delay(cfg, cond, seed, specs, out, timing) -> None:
    """RQ4: train [0, valid_day), valid [valid_day, deploy_day), test [deploy_day, end)."""
    t = time.perf_counter()
    ds = _dataset(cfg, cond, seed)
    _tick(timing, "generate", time.perf_counter() - t)
    p = cfg["split_days"]
    start = datetime.fromisoformat(ds.config.start_date)
    split = TemporalSplit.from_days(start, p["valid"], p["test"], ds.config.n_days)
    t = time.perf_counter()
    md = build_model_data(ds, LabelView(split.label_cutoff, cond["label_mode"]), "tabular+graph")
    _tick(timing, "features", time.perf_counter() - t)
    train = _rows(md, start, split.start, split.valid_start)
    valid = _rows(md, start, split.valid_start, split.test_start)
    test = _rows(md, start, split.test_start, split.end)
    for spec in specs:
        r = _fit_eval(spec, md, train, valid, test, seed)
        _tick(timing, f"model/{spec['label']}", r["fit_seconds"] + r["predict_seconds"])
        _append(
            out,
            _base_row(cfg, cond, seed, ds)
            | r
            | {
                "n_train": len(train),
                "n_test": len(test),
                "n_train_pos": int(md.y_train[train].sum()),
                "n_train_true_pos": int(md.y_true[train].sum()),
                "rss_gb": _rss_gb(),
            },
        )
        print(
            f"[{cfg['name']}] {cond} seed={seed} {spec['label']:22s} "
            f"auprc={r['metrics']['auprc']:.3f} train_pos={int(md.y_train[train].sum())}",
            flush=True,
        )


def _run_drift(cfg, cond, seed, specs, out, timing) -> None:
    """RQ2: static model deployed at ``deploy``; optional weekly retraining.

    Scores per evaluation window and over the pre/post/recovery segments are stored.
    Retrained models reuse the static model's class and params; each retrain at day
    ``r`` trains on ``[0, r - valid_days)``, validates on ``[r - valid_days, r)``, and
    serves days ``[r, r + every)``. Labels are oracle (RQ2 isolates drift from delay).
    """
    t = time.perf_counter()
    ds = _dataset(cfg, cond, seed)
    _tick(timing, "generate", time.perf_counter() - t)
    d = cfg["drift_protocol"]
    start = datetime.fromisoformat(ds.config.start_date)
    day = lambda x: start + timedelta(days=x)  # noqa: E731
    n_days = ds.config.n_days
    deploy, vdays = d["deploy_day"], d["valid_days"]
    md0 = build_model_data(ds, LabelView(day(deploy), "oracle"), "tabular+graph")
    test = _rows(md0, start, day(deploy), day(n_days))
    for spec in specs:
        train = _rows(md0, start, day(0), day(deploy - vdays))
        valid = _rows(md0, start, day(deploy - vdays), day(deploy))
        r = _fit_eval(spec, md0, train, valid, test, seed)
        static_scores = r.pop("_scores")
        rows = {"static": static_scores}
        if spec.get("retrain"):
            composite = static_scores.copy()
            for rd in range(deploy + d["retrain_every"], n_days, d["retrain_every"]):
                md_r = build_model_data(ds, LabelView(day(rd), "oracle"), "tabular+graph")
                tr = _rows(md_r, start, day(0), day(rd - vdays))
                va = _rows(md_r, start, day(rd - vdays), day(rd))
                serve = _rows(md_r, start, day(rd), day(min(rd + d["retrain_every"], n_days)))
                rr = _fit_eval(
                    spec | {"params": spec.get("retrain_params", spec["params"])},
                    md_r,
                    tr,
                    va,
                    serve,
                    seed,
                )
                # md_r rows are the same rows as md0 (same dataset, same order)
                composite[np.searchsorted(test, serve)] = rr["_scores"]
            rows["retrain"] = composite
        for policy, scores in rows.items():
            windows = _window_metrics(md0, test, scores, d, deploy, n_days)
            _append(
                out,
                _base_row(cfg, cond, seed, ds)
                | {k: v for k, v in r.items() if k not in ("metrics",)}
                | {
                    "model": f"{spec['label']}" + ("" if policy == "static" else "+retrain"),
                    "policy": policy,
                    "metrics": windows["overall"],
                    "windows": windows,
                    "rss_gb": _rss_gb(),
                },
            )
        _tick(timing, f"model/{spec['label']}", r["fit_seconds"] + r["predict_seconds"])
        print(
            f"[{cfg['name']}] seed={seed} {spec['label']:16s} "
            f"pre={windows['segments']['pre']['auprc']:.3f} "
            f"post={windows['segments']['post']['auprc']:.3f}",
            flush=True,
        )


def _window_metrics(md, test, scores, d, deploy, n_days) -> dict[str, Any]:
    y, amt, dayv = md.y_true[test], md.amount[test], md.day[test]

    def seg(a: float, b: float) -> dict[str, float]:
        m = (dayv >= a) & (dayv < b)
        if y[m].sum() == 0:
            return {"auprc": float("nan"), "n_fraud": 0}
        r = evaluate(y[m], scores[m], amt[m], dayv[m])
        return {
            "auprc": r["auprc"],
            "roc_auc": r["roc_auc"],
            "recall_at_k": r["recall_at_k"],
            "n_fraud": int(y[m].sum()),
        }

    w = d["window_days"]
    weekly = [
        {"start": a, "end": min(a + w, n_days), **seg(a, min(a + w, n_days))}
        for a in range(deploy, n_days, w)
    ]
    drift_day = d["drift_day"]
    rec = d["recovery_days"]
    return {
        "overall": evaluate(y, scores, amt, dayv),
        "weekly": weekly,
        "segments": {
            "pre": seg(deploy, drift_day),
            "post": seg(drift_day, n_days),
            "recovery": seg(rec[0], rec[1]),
        },
    }


def config_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
