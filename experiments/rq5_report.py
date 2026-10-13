"""RQ5 report: cost and investigator capacity, from the RQ1 runs at 0.5% fraud."""

from __future__ import annotations

import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from _common import (
    INK,
    INK_2,
    MODEL_ORDER,
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
    h5a = (
        "refuted"
        if G["ci_high"] < 0.10
        else ("supported" if G["ci_low"] >= 0.10 else "inconclusive")
    )

    seeds = sorted(set.intersection(*(set(per[m]["auprc"]) for m in models)))
    taus = []
    for s in seeds:
        a = [per[m]["auprc"][s] for m in models]
        c = [per[m][COST][s] for m in models]
        taus.append(float(kendalltau(a, c).statistic))
    T = stat(taus)
    h5b = "supported" if T["mean"] >= 0.8 else "refuted"

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in models:
        x, y = stat(per[m]["auprc"])["mean"], stat(per[m][COST])["mean"]
        ax.plot(x, y, "o", color=PALETTE[0], markersize=8, markeredgecolor="white")
        ax.annotate(
            label(m), (x, y), textcoords="offset points", xytext=(6, 3), fontsize=8, color=INK
        )
    ax.set_xlabel("Test AUPRC", color=INK_2)
    ax.set_ylabel("Recovered $ per investigator-hour", color=INK_2)
    style_axes(ax)
    ax.set_title(
        f"RQ5: ranking quality vs. operational value (k={cm.alerts_per_day}/day)",
        loc="left",
        fontsize=11,
    )
    save(fig, "rq5_cost")

    write_summary(
        "rq5",
        {
            "per_model": {m: {k: stat(v) for k, v in per[m].items()} for m in models},
            "best_model": best,
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
