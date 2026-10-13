"""Evaluation metrics: ranking, alert-capacity, threshold, calibration and cost."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class CostModel:
    """Operational cost model (DECISIONS D011, D021).

    A missed fraud costs its amount plus a chargeback fee. Every alert costs
    ``minutes_per_alert`` of investigator time at ``hourly_rate``. Investigators
    review at most ``alerts_per_day`` of the highest-ranked transactions per day.
    """

    chargeback_fee: float = 25.0
    minutes_per_alert: float = 10.0
    hourly_rate: float = 40.0
    alerts_per_day: int = 20

    @property
    def cost_per_alert(self) -> float:
        return self.hourly_rate * self.minutes_per_alert / 60


def daily_top_k(score: np.ndarray, day: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the ``k`` highest-scoring rows within each day (ties by index)."""
    order = np.lexsort((-score, day))
    d_sorted = day[order]
    first = np.r_[0, np.flatnonzero(np.diff(d_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
    alert = np.zeros(len(score), bool)
    alert[order[rank < k]] = True
    return alert


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(y))


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 (chosen on validation, applied to test)."""
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / max(ys.sum(), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    i = int(np.argmax(f1))
    return float(score[order][i])


def evaluate(
    y: np.ndarray,
    score: np.ndarray,
    amount: np.ndarray,
    day: np.ndarray,
    threshold: float | None = None,
    cost: CostModel | None = None,
    probabilistic: bool = True,
) -> dict[str, float]:
    """All test metrics for one run. ``score`` is p(fraud) or an anomaly score."""
    cost = cost or CostModel()
    out: dict[str, float] = {
        "auprc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "prevalence": float(y.mean()),
    }
    alert = daily_top_k(score, day, cost.alerts_per_day)
    tp = (alert & (y == 1)).sum()
    out["precision_at_k"] = float(tp / max(alert.sum(), 1))
    out["recall_at_k"] = float(tp / max(y.sum(), 1))
    out["n_alerts"] = float(alert.sum())

    loss = amount + cost.chargeback_fee
    caught = (alert & (y == 1)) * loss
    missed = ((~alert) & (y == 1)) * loss
    hours = alert.sum() * cost.minutes_per_alert / 60
    out["recovered_loss"] = float(caught.sum())
    out["recovered_per_investigator_hour"] = float(caught.sum() / max(hours, 1e-9))
    out["cost_total"] = float(missed.sum() + alert.sum() * cost.cost_per_alert)

    # Same capacity, ranking by expected loss p * amount (RQ5 policy ii).
    alert_el = daily_top_k(score * amount, day, cost.alerts_per_day) if probabilistic else alert
    caught_el = (alert_el & (y == 1)) * loss
    out["recovered_per_investigator_hour_el"] = float(
        caught_el.sum() / max(alert_el.sum() * cost.minutes_per_alert / 60, 1e-9)
    )

    if threshold is not None:
        out["f1_at_valid_threshold"] = float(f1_score(y, score >= threshold, zero_division=0))
    out["ece"] = ece(y, np.clip(score, 0, 1)) if probabilistic else float("nan")
    return out
