"""RQ4 report: label delay."""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory as blend

from _common import (
    INK,
    INK_2,
    LINE,
    PALETTE,
    by_seed,
    figure,
    fmt,
    label,
    legend_above,
    load_runs,
    md_table,
    save,
    stat,
    style_axes,
    write_summary,
    write_table,
)

SUPERVISED = ["logreg", "lgbm_tuned", "gru"]
# Label condition is the variable here, not the model: ink for the oracle, orange for delay.
ORACLE, DELAYED = INK, PALETTE[1]


def validation_positives(runs: list[dict]) -> dict[str, dict]:
    """Frauds visible in the validation window [day 50, day 60) at deployment (day 60).

    Recomputed from the generator (deterministic in the seed), so the paper can
    report how much validation signal each label mode leaves.
    """
    from datetime import datetime, timedelta

    import polars as pl

    from fraudlab.features import LabelView, training_labels
    from fraudlab.synth import generate, preset

    out: dict[str, list[int]] = {"oracle": [], "delayed": [], "true": []}
    for seed in sorted({r["seed"] for r in runs}):
        ds = generate(preset("small_long", seed=seed, fraud_rate=0.005))
        start = datetime.fromisoformat(ds.config.start_date)
        lo, hi = start + timedelta(days=50), start + timedelta(days=60)
        v = ds.transactions.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
        v = v.select("tx_id").join(ds.labels.select("tx_id", "is_fraud"), on="tx_id")
        out["true"].append(int(v["is_fraud"].sum()))
        for mode in ("oracle", "delayed"):
            y = training_labels(ds, LabelView(hi, mode)).join(v.select("tx_id"), on="tx_id")
            out[mode].append(int(y["y_train"].sum()))
    return {
        k: {"per_seed": v, "min": min(v), "max": max(v), "seeds_with_zero": sum(x == 0 for x in v)}
        for k, v in out.items()
    }


MODELS = [*SUPERVISED, "lgbm_semi"]


def main() -> None:
    runs = load_runs("rq4")
    au = {
        m: {mode: by_seed(runs, m, "auprc", label_mode=mode) for mode in ("oracle", "delayed")}
        for m in MODELS
    }
    pos = {
        mode: stat(
            [
                r["n_train_pos"]
                for r in runs
                if r["condition"]["label_mode"] == mode and r["model"] == "lgbm_tuned"
            ]
        )
        for mode in ("oracle", "delayed")
    }
    loss, rows = {}, []
    for m in MODELS:
        o, d = au[m]["oracle"], au[m]["delayed"]
        loss[m] = stat({s: 1 - d[s] / o[s] for s in o if s in d and o[s] > 0})
        rows.append(
            [
                label(m),
                fmt(stat(o)),
                fmt(stat(d)),
                fmt(loss[m], 2),
                f"[{loss[m]['ci_low']:.2f}, {loss[m]['ci_high']:.2f}]",
            ]
        )
    write_table(
        "rq4_delay",
        md_table(
            [
                "Model",
                "AUPRC oracle labels",
                "AUPRC delayed labels",
                "Relative loss",
                "95% CI of loss",
            ],
            rows,
        ),
    )

    h4a_refuting = [m for m in SUPERVISED if loss[m]["ci_high"] < 0.10]
    h4a = "refuted" if h4a_refuting else "not refuted"

    lo, ld, sd = au["lgbm_tuned"]["oracle"], au["lgbm_tuned"]["delayed"], au["lgbm_semi"]["delayed"]
    rec = {
        s: (sd[s] - ld[s]) / (lo[s] - ld[s])
        for s in lo
        if s in ld and s in sd and lo[s] - ld[s] > 0
    }
    R = stat(rec)
    gap = stat({s: lo[s] - ld[s] for s in lo if s in ld})
    h4b = "refuted" if (R["n"] == 0 or R["ci_high"] < 0.25) else "not refuted"
    write_table(
        "rq4_recovery",
        md_table(
            ["Quantity", "Value"],
            [
                ["Training positives visible (oracle)", fmt(pos["oracle"], 0)],
                ["Training positives visible (delayed)", fmt(pos["delayed"], 0)],
                ["AUPRC lost by LightGBM (oracle − delayed)", fmt(gap)],
                [
                    "Recovery by LightGBM + anomaly scores",
                    fmt(R, 2)
                    + f" (95% CI [{R['ci_low']:.2f}, {R['ci_high']:.2f}],"
                    + f" seeds with a loss: {R['n']})",
                ],
            ],
        ),
    )

    # Figure: a dumbbell per model from delayed to oracle labels, the relative loss at the right.
    fig, ax = figure(width=0.85, height=2.1)
    ys = np.arange(len(MODELS))[::-1]
    n_seeds = max(stat(au[m]["oracle"])["n"] for m in MODELS)
    for y, m in zip(ys, MODELS, strict=True):
        o, d = stat(au[m]["oracle"]), stat(au[m]["delayed"])
        ax.plot([d["mean"], o["mean"]], [y, y], color=LINE, linewidth=1.2, zorder=1)
        for value, fill in ((o["mean"], ORACLE), (d["mean"], DELAYED)):
            ax.plot(
                value,
                y,
                "o",
                color=fill,
                markersize=4.6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )
        ax.text(
            1.0,
            y,
            f"−{loss[m]['mean']:.0%}",
            transform=blend(ax.transAxes, ax.transData),
            ha="left",
            va="center",
            fontsize=7.5,
            color=INK_2,
        )
    ax.set_yticks(ys, [label(m) for m in MODELS])
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_xlim(0, 0.8)
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.set_xlabel(f"Test AUPRC (mean of {n_seeds} seeds)")
    style_axes(ax, grid="x", left=False)
    legend_above(
        ax,
        [
            Line2D([], [], marker="o", linestyle="none", markersize=4.6, color=ORACLE, label=t)
            for t in ("Oracle labels",)
        ]
        + [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4.6,
                color=DELAYED,
                label="Delayed labels (7–30 d)",
            )
        ],
        ncols=2,
    )
    save(fig, "rq4_delay")

    write_summary(
        "rq4",
        {
            "auprc": {m: {k: stat(v) for k, v in au[m].items()} for m in MODELS},
            "relative_loss": loss,
            "recovery": R,
            "lgbm_gap": gap,
            "train_positives": pos,
            "hypotheses": {"H4a": h4a, "H4b": h4b},
            "h4a_refuting_models": h4a_refuting,
            "validation_positives": validation_positives(runs),
        },
    )
    print("RQ4", {"H4a": h4a, "H4b": h4b, "refuting": h4a_refuting, "R": R["mean"]})


if __name__ == "__main__":
    main()
