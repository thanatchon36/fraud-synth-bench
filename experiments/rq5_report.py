"""RQ5 report: cost and investigator capacity, from the RQ1 runs at 0.5% fraud."""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import kendalltau

from _common import (
    INK,
    INK_2,
    MODEL_ORDER,
    by_seed,
    figure,
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
from fraudlab.eval import CostModel

RATE = 0.005
COST = "recovered_per_investigator_hour"


def main() -> None:
    runs = load_runs("rq1")
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in runs)]
    prob_models = [m for m in models if m not in ("iforest", "autoencoder")]
    per = {
        m: {
            k: by_seed(runs, m, k, fraud_rate=RATE)
            for k in ("auprc", COST, f"{COST}_el", "cost_total")
        }
        for m in models
    }

    rows = []
    for m in models:
        el = fmt(stat(per[m][f"{COST}_el"]), 0) if m in prob_models else "n/a (score)"
        rows.append(
            [
                label(m),
                fmt(stat(per[m]["auprc"])),
                fmt(stat(per[m][COST]), 0),
                el,
                fmt(stat(per[m]["cost_total"]), 0),
            ]
        )
    cm = CostModel()
    write_table(
        "rq5_cost",
        md_table(
            [
                "Model (0.5% fraud)",
                "AUPRC",
                "Recovered $ / investigator-hour (rank by p)",
                "… (rank by p × amount)",
                f"Total cost $ (missed + {cm.cost_per_alert:.2f}/alert)",
            ],
            rows,
        ),
    )

    best = max(prob_models, key=lambda m: stat(per[m][COST])["mean"])
    gain = {
        s: per[best][f"{COST}_el"][s] / per[best][COST][s] - 1
        for s in per[best][COST]
        if per[best][COST][s] > 0
    }
    G = stat(gain)
    h5a = "refuted" if G["ci_high"] < 0.10 else "not refuted"

    seeds = sorted(set.intersection(*(set(per[m]["auprc"]) for m in models)))
    taus = []
    for s in seeds:
        a = [per[m]["auprc"][s] for m in models]
        c = [per[m][COST][s] for m in models]
        taus.append(float(kendalltau(a, c).statistic))
    T = stat(taus)
    h5b = "not refuted" if T["mean"] >= 0.8 else "refuted"

    fig, ax = figure(width=0.8, height=2.9)
    for m in models:
        x, y = stat(per[m]["auprc"])["mean"], stat(per[m][COST])["mean"]
        ax.plot(
            x,
            y,
            "o",
            color=INK,
            markersize=4.6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        # Below-left for the model that sits on top of tuned LightGBM.
        below = m == "random_forest"
        ax.annotate(
            label(m),
            (x, y),
            textcoords="offset points",
            xytext=(-5, -9) if below else (5, 3),
            ha="right" if below else "left",
            fontsize=7.5,
            color=INK,
        )
    ax.text(
        0.98,
        0.04,
        f"Kendall τ between the two rankings: {T['mean']:.2f} (mean of {T['n']} seeds)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK_2,
    )
    ax.set_xlim(0, 0.9)
    ax.set_xticks(np.arange(0, 0.91, 0.2))
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Test AUPRC (mean of 5 seeds)".replace("5", str(T["n"])))
    ax.set_ylabel("Recovered loss ($) per investigator-hour")
    style_axes(ax, grid="y")
    save(fig, "rq5_cost")

    # Business view: share of fraud losses recovered within capacity.
    def share(m: str) -> dict[str, float]:
        vals = {}
        for r in runs:
            if r["model"] == m and r["condition"]["fraud_rate"] == RATE:
                x = r["metrics"]
                total = x["recovered_loss"] + x["cost_total"] - x["n_alerts"] * cm.cost_per_alert
                vals[r["seed"]] = x["recovered_loss"] / total if total > 0 else float("nan")
        return stat(vals)

    loss_share = {m: share(m) for m in models}
    hours_per_day = cm.alerts_per_day * cm.minutes_per_alert / 60

    write_summary(
        "rq5",
        {
            "per_model": {m: {k: stat(v) for k, v in per[m].items()} for m in models},
            "best_model": best,
            "loss_recovered_share": loss_share,
            "investigator_hours_per_day": hours_per_day,
            "el_gain": G,
            "kendall_tau": T,
            "hypotheses": {"H5a": h5a, "H5b": h5b},
            "cost_model": {
                "alerts_per_day": cm.alerts_per_day,
                "minutes_per_alert": cm.minutes_per_alert,
                "chargeback_fee": cm.chargeback_fee,
                "hourly_rate": cm.hourly_rate,
            },
        },
    )
    print("RQ5", {"H5a": h5a, "H5b": h5b, "best": best, "gain": G["mean"], "tau": T["mean"]})


if __name__ == "__main__":
    main()
