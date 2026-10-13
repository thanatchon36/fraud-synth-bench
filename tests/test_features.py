"""Feature builders must use only the past (no leakage)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from functools import cache

import numpy as np
import polars as pl
import pytest

from fraudlab.features import GRAPH, TABULAR, LabelView, build_features, training_labels
from fraudlab.synth import SyntheticDataset, generate, preset

CUTOFF = datetime(2025, 1, 19)


@cache
def ds() -> SyntheticDataset:
    return generate(preset("tiny", seed=0, fraud_rate=0.02))


def truncate(d: SyntheticDataset, at: datetime) -> SyntheticDataset:
    """The dataset as it would look at time ``at``: no rows, no labels from the future."""
    tx = d.transactions.filter(pl.col("timestamp") < at)
    labels = d.labels.join(tx.select("tx_id"), on="tx_id").with_columns(
        pl.when(pl.col("label_available_at") < at)
        .then(pl.col("label_available_at"))
        .otherwise(None)
        .alias("label_available_at"),
        pl.when(pl.col("label_available_at") < at)
        .then(pl.col("observed_label"))
        .otherwise(0)
        .alias("observed_label"),
    )
    return replace(d, transactions=tx, labels=labels)


def test_features_ignore_the_future() -> None:
    at = datetime(2025, 1, 21, 13, 0)
    full = build_features(ds(), LabelView(CUTOFF)).filter(pl.col("timestamp") < at)
    past = build_features(truncate(ds(), at), LabelView(CUTOFF))
    assert full.height == past.height
    cols = list(TABULAR) + list(GRAPH)
    a = full.sort("tx_id").select(cols).to_numpy()
    b = past.sort("tx_id").select(cols).to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-6, equal_nan=True)


def test_own_label_does_not_change_own_features() -> None:
    f0 = build_features(ds(), LabelView(CUTOFF))
    fraud_id = ds().labels.filter(
        (pl.col("observed_label") == 1) & (pl.col("label_available_at") < CUTOFF)
    )["tx_id"][0]
    flipped = ds().labels.with_columns(
        pl.when(pl.col("tx_id") == fraud_id)
        .then(0)
        .otherwise(pl.col("observed_label"))
        .alias("observed_label")
    )
    f1 = build_features(replace(ds(), labels=flipped), LabelView(CUTOFF))
    row = lambda f: f.filter(pl.col("tx_id") == fraud_id).select(list(GRAPH) + list(TABULAR))  # noqa: E731
    assert row(f0).equals(row(f1))


def test_rolling_counts_match_brute_force() -> None:
    f = build_features(ds(), LabelView(CUTOFF))
    tx = ds().transactions
    rng = np.random.default_rng(0)
    for i in rng.choice(f.height, 40, replace=False):
        r = f.row(int(i), named=True)
        hist = tx.filter(
            (pl.col("account_id") == r["account_id"]) & (pl.col("timestamp") < r["timestamp"])
        )
        in24 = hist.filter(pl.col("timestamp") > r["timestamp"] - timedelta(hours=24))
        assert r["acc_cnt_24h"] == in24.height
        assert r["acc_hist_cnt"] >= hist.height  # ties at equal timestamps count by order
        assert r["acc_sum_24h"] == pytest.approx(in24["amount"].sum(), rel=1e-4)


def test_label_view_modes() -> None:
    oracle = training_labels(ds(), LabelView(CUTOFF, "oracle"))
    delayed = training_labels(ds(), LabelView(CUTOFF, "delayed"))
    lab = ds().labels.join(ds().transactions.select("tx_id", "timestamp"), on="tx_id")
    in_train = lab.filter(pl.col("timestamp") < CUTOFF)["tx_id"].implode()
    n_oracle = oracle.filter(pl.col("tx_id").is_in(in_train))["y_train"].sum()
    n_delayed = delayed.filter(pl.col("tx_id").is_in(in_train))["y_train"].sum()
    detected = lab.filter((pl.col("timestamp") < CUTOFF) & (pl.col("observed_label") == 1))
    assert n_oracle == detected.height
    assert 0 < n_delayed < n_oracle  # delay hides recent fraud
    # nothing after the cutoff is ever labelled
    assert oracle.join(lab, on="tx_id").filter(pl.col("timestamp") >= CUTOFF)["y_train"].sum() == 0
