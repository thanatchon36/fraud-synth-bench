"""Shared helpers for the per-RQ report scripts.

Each report script reads ``results/runs/<rq>.jsonl``, writes Markdown tables to
``results/tables/``, figures to ``results/figures/`` and its slice of the summary
to ``results/summary/<rq>.json``. ``build_summary.py`` merges the slices into
``results/summary.json``, which the paper template reads.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fraudlab.eval import mean_ci

RESULTS = Path("results")
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
SUMMARY = RESULTS / "summary"

# Reference categorical palette (dataviz skill, light mode), fixed order. Colour
# follows the entity: each model keeps its slot in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_ORDER = [
    "lgbm_tuned",
    "lgbm_default",
    "random_forest",
    "logreg",
    "gru",
    "ft_transformer",
    "tabpfn",
    "iforest",
    "autoencoder",
    "graphsage",
    "lgbm_tuned+graph",
    "lgbm_semi",
]
MODEL_LABEL = {
    "lgbm_tuned": "LightGBM (tuned)",
    "lgbm_default": "LightGBM (default)",
    "random_forest": "Random forest",
    "logreg": "Logistic regression",
    "gru": "GRU",
    "ft_transformer": "FT-Transformer",
    "tabpfn": "TabPFN",
    "iforest": "Isolation Forest",
    "autoencoder": "Autoencoder",
    "graphsage": "GraphSAGE",
    "lgbm_tuned+graph": "LightGBM + graph features",
    "lgbm_semi": "LightGBM + anomaly scores",
}


def label(model: str) -> str:
    for suffix, text in (("+retrain", " + weekly re-tune"), ("+refit", " + weekly refit")):
        if model.endswith(suffix):
            base = model.removesuffix(suffix)
            return MODEL_LABEL.get(base, base) + text
    return MODEL_LABEL.get(model, model)


def load_runs(name: str) -> list[dict[str, Any]]:
    path = RESULTS / "runs" / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found: run `fraudlab train --config experiments/configs/{name}.yaml`"
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_seed(runs: list[dict], model: str, metric: str, **cond: Any) -> dict[int, float]:
    out = {}
    for r in runs:
        if r["model"] == model and all(r["condition"].get(k) == v for k, v in cond.items()):
            out[r["seed"]] = r["metrics"][metric]
    return out


def paired_arrays(a: dict[int, float], b: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    seeds = sorted(set(a) & set(b))
    return np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])


def stat(values: dict[int, float] | list[float]) -> dict[str, float]:
    x = np.array(list(values.values()) if isinstance(values, dict) else values, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": math.nan, "std": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    m, s, lo, hi = mean_ci(x)
    return {"mean": m, "std": s, "ci_low": lo, "ci_high": hi, "n": len(x)}


def fmt(s: dict[str, float], digits: int = 3) -> str:
    if s["n"] == 0 or math.isnan(s["mean"]):
        return "–"
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_table(name: str, text: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.md").write_text(text)


def write_summary(name: str, data: dict[str, Any]) -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / f"{name}.json").write_text(json.dumps(_clean(data), indent=2, sort_keys=True))


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, np.generic):
        return _clean(x.item())
    return x


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x" if ax.get_ylabel() == "" else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


#: Explicit slot per model that appears in multi-series charts (unique, never cycled).
COLOR_SLOT = {
    "lgbm_tuned": 0,
    "logreg": 1,
    "gru": 2,
    "iforest": 3,
    "autoencoder": 4,
    "ft_transformer": 5,
    "graphsage": 6,
    "lgbm_tuned+graph": 7,
}


def color(model: str) -> str:
    """Stable colour per model (follows the entity, never its rank)."""
    base = model.removesuffix("+retrain").removesuffix("+refit")
    return PALETTE[COLOR_SLOT[base]] if base in COLOR_SLOT else INK_2


def dot_plot(
    ax: plt.Axes, rows: list[tuple[str, dict[str, float]]], xlabel: str, ref: float | None = None
) -> None:
    """Horizontal mean ± 95% CI dot plot; one row per model, single neutral colour."""
    ys = np.arange(len(rows))[::-1]
    for y, (_name, s) in zip(ys, rows, strict=True):
        if s["n"] == 0:
            continue
        ax.plot(
            [s["ci_low"], s["ci_high"]], [y, y], color=INK_2, linewidth=2, solid_capstyle="round"
        )
        ax.plot(
            s["mean"],
            y,
            "o",
            color=PALETTE[0],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
    ax.set_yticks(ys, [label(n) for n, _ in rows], color=INK)
    ax.set_xlabel(xlabel, color=INK_2)
    if ref is not None:
        ax.axvline(ref, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    style_axes(ax)


def pretty_verdict(verdict: str, a: str, b: str) -> str:
    """Replace model ids in a verdict with display names (longest id first)."""
    for m in sorted({a, b}, key=len, reverse=True):
        verdict = verdict.replace(m, f"\x00{MODEL_ORDER.index(m) if m in MODEL_ORDER else m}\x00")
    for m in {a, b}:
        key = f"\x00{MODEL_ORDER.index(m) if m in MODEL_ORDER else m}\x00"
        verdict = verdict.replace(key, label(m))
    return verdict
