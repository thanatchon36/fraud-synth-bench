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
from matplotlib import font_manager
from matplotlib.lines import Line2D

from fraudlab.eval import mean_ci

RESULTS = Path("results")
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
SUMMARY = RESULTS / "summary"

# House style for the five figures, chosen so they sit inside the typeset paper rather than
# beside it:
#
# * Width is the PDF text block (A4 minus pandoc's 1.25 in side margins) times the fraction at
#   which the paper displays the figure, so text drawn at 8 pt prints at 8 pt. Nothing is scaled.
# * Libertinus Serif, the body face, vendored in experiments/fonts (SIL OFL), so the output does
#   not depend on the machine's fonts.
# * No title in the image: the caption names the figure. Legends sit above the plotting area,
#   never on the data.
# * Colour follows the entity: each model keeps its slot in every figure (reference categorical
#   palette, fixed order); the single-series dot plots use ink.
# * SVG with text as outlines for the PDF; PNG at 300 dpi for the web reader and README.
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FAMILY = "Libertinus Serif"
TEXT_WIDTH_IN = 210 / 25.4 - 2 * 1.25
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, RULE, GRID, LINE = "#1a1a1a", "#555555", "#3a3a3a", "#e4e4e4", "#a0a0a0"
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


def style() -> None:
    """Register the vendored typeface and set the house rcParams."""
    for path in sorted(FONT_DIR.glob("LibertinusSerif-*.otf")):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 8.5,
            "text.color": INK,
            "axes.labelsize": 8.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": RULE,
            "ytick.color": RULE,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.handlelength": 2.4,
            "legend.handleheight": 0.8,
            "legend.columnspacing": 1.4,
            "legend.borderaxespad": 0.2,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "svg.fonttype": "path",
            "svg.hashsalt": "fraud-synth-bench",
            "figure.dpi": 100,
            "savefig.facecolor": "white",
        }
    )


def figure(
    width: float = 1.0, height: float = 2.8, ncols: int = 1, **kwargs: Any
) -> tuple[plt.Figure, Any]:
    """Return a figure ``width`` text blocks wide, laid out by matplotlib's constrained engine."""
    style()
    fig, axes = plt.subplots(
        1, ncols, figsize=(TEXT_WIDTH_IN * width, height), layout="constrained", **kwargs
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06)
    return fig, axes


def style_axes(ax: plt.Axes, grid: str | None = "y", *, left: bool = True) -> None:
    """Open spines, outward ticks and a recessive grid along the quantitative axis only."""
    ax.spines["left"].set_visible(left)
    if not left:
        ax.tick_params(axis="y", length=0)
    if grid:
        ax.grid(axis=grid)
    ax.set_axisbelow(True)


def legend_above(ax: plt.Axes, handles: list[Any], ncols: int) -> None:
    """Put the legend in a row above the plotting area, left-aligned with it."""
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncols=ncols,
        borderaxespad=0.0,
    )


def series_handle(model: str, *, dashed: bool = False) -> Line2D:
    """Legend entry for one model's line: its colour slot, a marker, dashed for a refit policy."""
    return Line2D(
        [],
        [],
        color=color(model),
        marker="o",
        markersize=3.6,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linewidth=1.1,
        linestyle=(0, (4, 2)) if dashed else "-",
        label=label(model),
    )


def spread(positions: list[float], gap: float) -> list[float]:
    """Push sorted label positions apart until neighbours are at least ``gap`` apart."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    placed = [positions[i] for i in order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    shift = (sum(placed) - sum(positions[i] for i in order)) / len(placed)
    placed = [p - shift for p in placed]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    result = [0.0] * len(positions)
    for slot, i in enumerate(order):
        result[i] = placed[slot]
    return result


def save(fig: plt.Figure, name: str) -> None:
    """Write the SVG (vector, for the PDF) and the PNG (300 dpi, for the web) of one figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", metadata={"Date": None, "Creator": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=300, metadata={"Software": None})
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
    ax: plt.Axes,
    rows: list[tuple[str, dict[str, float]]],
    xlabel: str,
    *,
    separators: tuple[int, ...] = (),
) -> None:
    """Horizontal mean and 95% CI dot plot, one row per model, in ink.

    ``separators`` are row indices above which a thin rule is drawn, to set groups of models
    apart (the unsupervised references, say) without a legend.
    """
    ys = np.arange(len(rows))[::-1]
    for y, (_name, s) in zip(ys, rows, strict=True):
        if s["n"] == 0:
            continue
        ax.plot(
            [s["ci_low"], s["ci_high"]],
            [y, y],
            color=INK,
            linewidth=1.0,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            s["mean"],
            y,
            "o",
            color=INK,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
    for index in separators:
        ax.axhline(ys[index] + 0.5, color=RULE, linewidth=0.5, zorder=1)
    ax.set_yticks(ys, [label(n) for n, _ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel)
    style_axes(ax, grid="x", left=False)


def pretty_verdict(verdict: str, a: str, b: str) -> str:
    """Replace model ids in a verdict with display names (longest id first)."""
    for m in sorted({a, b}, key=len, reverse=True):
        verdict = verdict.replace(m, f"\x00{MODEL_ORDER.index(m) if m in MODEL_ORDER else m}\x00")
    for m in {a, b}:
        key = f"\x00{MODEL_ORDER.index(m) if m in MODEL_ORDER else m}\x00"
        verdict = verdict.replace(key, label(m))
    return verdict
