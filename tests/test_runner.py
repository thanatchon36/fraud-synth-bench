"""The experiment runner end to end on tiny data (fast models only)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import fraudlab  # noqa: F401  (load order: LightGBM before torch)
from fraudlab.runner import run


def _write(tmp: Path, cfg: dict) -> Path:
    p = tmp / f"{cfg['name']}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _rows(tmp: Path, name: str) -> list[dict]:
    return [json.loads(x) for x in (tmp / "runs" / f"{name}.jsonl").read_text().splitlines()]


def test_standard_protocol_and_resume(tmp_path: Path) -> None:
    cfg = {
        "name": "t_std",
        "protocol": "standard",
        "dataset": {"preset": "tiny"},
        "conditions": {"fraud_rate": [0.01]},
        "seeds": [0, 1],
        "models": [
            "logreg",
            "lgbm_default",
            {"name": "lgbm_default", "features": "tabular+graph"},
            "iforest",
        ],
    }
    path = _write(tmp_path, cfg)
    run(path, root=tmp_path)
    rows = _rows(tmp_path, "t_std")
    assert len(rows) == 8
    assert {r["model"] for r in rows} == {"logreg", "lgbm_default", "lgbm_default+graph", "iforest"}
    for r in rows:
        m = r["metrics"]
        assert 0 <= m["auprc"] <= 1 and 0 <= m["roc_auc"] <= 1
        assert r["n_test_fraud"] > 0 and r["dataset_hash"]
    # same seed -> same dataset for every model (paired design)
    assert len({r["dataset_hash"] for r in rows if r["seed"] == 0}) == 1
    run(path, root=tmp_path)  # resume: nothing new
    assert len(_rows(tmp_path, "t_std")) == 8
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert "t_std" in timing


def test_drift_protocol(tmp_path: Path) -> None:
    cfg = {
        "name": "t_drift",
        "protocol": "drift",
        "dataset": {"preset": "tiny", "n_days": 90, "drift": "rq2", "fraud_rate": 0.01},
        "seeds": [0],
        "drift_protocol": {
            "deploy_day": 50,
            "valid_days": 10,
            "drift_day": 54,
            "window_days": 7,
            "retrain_every": 14,
            "recovery_days": [61, 68],
        },
        "models": [{"name": "lgbm_default", "retrain": True}, "iforest"],
    }
    run(_write(tmp_path, cfg), root=tmp_path)
    rows = _rows(tmp_path, "t_drift")
    assert {r["model"] for r in rows} == {"lgbm_default", "lgbm_default+retrain", "iforest"}
    for r in rows:
        assert set(r["windows"]["segments"]) == {"pre", "post", "recovery"}
        assert len(r["windows"]["weekly"]) == 6


def test_delay_protocol(tmp_path: Path) -> None:
    cfg = {
        "name": "t_delay",
        "protocol": "delay",
        "dataset": {"preset": "tiny", "n_days": 90, "fraud_rate": 0.01},
        "split_days": {"valid": 50, "test": 60},
        "conditions": {"label_mode": ["oracle", "delayed"]},
        "seeds": [0],
        "models": ["lgbm_default"],
    }
    run(_write(tmp_path, cfg), root=tmp_path)
    rows = {r["condition"]["label_mode"]: r for r in _rows(tmp_path, "t_delay")}
    assert rows["delayed"]["n_train_pos"] < rows["oracle"]["n_train_pos"]
    assert rows["oracle"]["n_train_pos"] <= rows["oracle"]["n_train_true_pos"]
