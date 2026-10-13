"""Pre-registered statistical protocol (docs/RESEARCH_QUESTIONS.md, DECISIONS D006)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

DELTA = 0.02  # practical-significance margin (absolute AUPRC)


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    """Mean, std (ddof=1), and t-based CI over seeds."""
    x = np.asarray(x, float)
    m, s = float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
    if len(x) < 2:
        return m, s, m, m
    h = stats.t.ppf((1 + level) / 2, len(x) - 1) * s / np.sqrt(len(x))
    return m, s, m - h, m + h


@dataclass
class Comparison:
    a: str
    b: str
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float = float("nan")
    wilcoxon_p: float = float("nan")
    verdict: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def paired(a_name: str, a: np.ndarray, b_name: str, b: np.ndarray) -> Comparison:
    """Paired comparison over seeds of metric(A) - metric(B)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m, _, lo, hi = mean_ci(d)
    if np.allclose(d, d[0]):
        p = 0.0 if abs(d[0]) > 0 else 1.0
    else:
        p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w = float("nan")
    return Comparison(a_name, b_name, m, lo, hi, p, wilcoxon_p=w)


def holm(comparisons: list[Comparison], delta: float = DELTA) -> list[Comparison]:
    """Holm–Bonferroni adjustment, then the three-way verdict for each comparison."""
    k = len(comparisons)
    order = np.argsort([c.p_value for c in comparisons])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (k - rank) * comparisons[i].p_value)
        running = max(running, adj)
        comparisons[i].p_holm = running
    for c in comparisons:
        c.verdict = verdict(c, delta)
    return comparisons


def verdict(c: Comparison, delta: float = DELTA) -> str:
    """``A beats B`` / ``B beats A`` / ``equivalent`` / ``inconclusive``."""
    if c.p_holm < 0.05 and c.mean_diff >= delta:
        return f"{c.a} beats {c.b}"
    if c.p_holm < 0.05 and c.mean_diff <= -delta:
        return f"{c.b} beats {c.a}"
    if c.ci_low > -delta and c.ci_high < delta:
        return "equivalent"
    return "inconclusive"
