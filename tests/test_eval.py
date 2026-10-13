"""Metrics, splits and the statistical protocol."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from fraudlab.eval import (
    CostModel,
    TemporalSplit,
    best_f1_threshold,
    daily_top_k,
    ece,
    evaluate,
    holm,
    paired,
)


def test_split_fractions_are_ordered_day_boundaries() -> None:
    s = TemporalSplit.from_fractions(datetime(2025, 1, 1), 30)
    assert s.start < s.valid_start < s.test_start < s.end
    assert (s.valid_start - s.start).days == 18
    assert (s.test_start - s.start).days == 22  # round(30 * 0.75)
    assert s.label_cutoff == s.test_start


def test_daily_top_k() -> None:
    score = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.6])
    day = np.array([0, 0, 0, 1, 1, 1])
    alert = daily_top_k(score, day, k=2)
    assert alert.tolist() == [True, False, True, True, False, True]


def test_perfect_and_random_scores() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.01).astype(int)
    amount = rng.uniform(1, 100, 5000)
    day = np.repeat(np.arange(50), 100)
    perfect = evaluate(y, y.astype(float), amount, day)
    assert perfect["auprc"] == 1.0 and perfect["roc_auc"] == 1.0
    noise = evaluate(y, rng.random(5000), amount, day)
    assert noise["auprc"] < 0.05
    assert abs(noise["roc_auc"] - 0.5) < 0.1


def test_cost_metrics() -> None:
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.1, 0.0])
    amount = np.array([100.0, 5.0, 200.0, 5.0])
    day = np.zeros(4, int)
    cm = CostModel(chargeback_fee=25, minutes_per_alert=10, hourly_rate=60, alerts_per_day=2)
    m = evaluate(y, score, amount, day, cost=cm)
    assert m["recovered_loss"] == 125
    assert m["recovered_per_investigator_hour"] == pytest.approx(125 / (20 / 60))
    assert m["cost_total"] == pytest.approx(225 + 2 * 10)
    # ranking by p * amount alerts the 200 fraud instead of the 5 legit
    assert m["recovered_per_investigator_hour_el"] > m["recovered_per_investigator_hour"]


def test_ece_and_threshold() -> None:
    y = np.array([0, 0, 1, 1])
    assert ece(y, y.astype(float)) == 0.0
    assert ece(y, np.full(4, 0.5)) == 0.0  # constant 0.5 on a 50% base rate is calibrated
    assert ece(y, np.full(4, 0.9)) == pytest.approx(0.4)
    assert best_f1_threshold(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.3])) == 0.8


def test_paired_and_holm_verdicts() -> None:
    base = np.array([0.50, 0.52, 0.49, 0.51, 0.50])
    better = base + np.array([0.06, 0.05, 0.07, 0.06, 0.05])
    same = base + np.array([0.001, -0.001, 0.0005, 0.0, -0.0005])
    noisy = base + np.array([0.10, -0.12, 0.08, -0.09, 0.02])
    cs = holm(
        [
            paired("A", better, "B", base),
            paired("C", same, "B", base),
            paired("D", noisy, "B", base),
        ]
    )
    assert cs[0].verdict == "A beats B"
    assert cs[1].verdict == "equivalent"
    assert cs[2].verdict == "inconclusive"
    assert all(c.p_holm >= c.p_value for c in cs)
