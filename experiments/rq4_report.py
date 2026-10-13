"""RQ4 report: label delay."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    INK_2,
    PALETTE,
    by_seed,
    fmt,
    label,
    load_runs,
    md_table,
    save,
    stat,
    style_axes,
    write_summary,
    write_table,
)

SUPERVISED = ["logreg", "lgbm_tuned", "gru"]


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

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ys = np.arange(len(MODELS))[::-1]
    for y, m in zip(ys, MODELS, strict=True):
        o, d = stat(au[m]["oracle"]), stat(au[m]["delayed"])
        ax.plot([d["mean"], o["mean"]], [y, y], color=INK_2, linewidth=2)
        ax.plot(
            o["mean"],
            y,
            "o",
            color=PALETTE[0],
            markersize=8,
            markeredgecolor="white",
            label="Oracle labels" if y == ys[0] else None,
        )
        ax.plot(
            d["mean"],
            y,
            "o",
            color=PALETTE[1],
            markersize=8,
            markeredgecolor="white",
            label="Delayed labels (7–30 d)" if y == ys[0] else None,
        )
    ax.set_yticks(ys, [label(m) for m in MODELS])
    ax.set_xlabel("Test AUPRC (mean of 5 seeds)", color=INK_2)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("RQ4: effect of label delay (small_long preset)", loc="left", fontsize=11)
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
