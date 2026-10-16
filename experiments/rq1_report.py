"""RQ1 report: old vs. new under extreme imbalance.

Inputs: results/runs/rq1.jsonl. Outputs: results/tables/rq1_*.md,
results/figures/rq1_auprc.png, results/summary/rq1.json.
"""

from __future__ import annotations

import math

import numpy as np

from _common import (
    INK_2,
    MODEL_ORDER,
    by_seed,
    dot_plot,
    figure,
    fmt,
    label,
    load_runs,
    md_table,
    paired_arrays,
    pretty_verdict,
    save,
    stat,
    write_summary,
    write_table,
)
from fraudlab.eval import holm, paired

RATES = [0.001, 0.005, 0.01]
MODERN = ["ft_transformer", "tabpfn"]
UNSUPERVISED = ["iforest", "autoencoder"]


def main() -> None:
    runs = load_runs("rq1")
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in runs)]

    # Main table: AUPRC by fraud rate.
    rows, auprc = [], {}
    for m in models:
        auprc[m] = {rate: stat(by_seed(runs, m, "auprc", fraud_rate=rate)) for rate in RATES}
        rows.append([label(m)] + [fmt(auprc[m][r]) for r in RATES])
    write_table("rq1_auprc", md_table(["Model"] + [f"AUPRC @ {r:.1%} fraud" for r in RATES], rows))

    # Secondary metrics at 0.5%.
    sec = [
        "roc_auc",
        "precision_at_k",
        "recall_at_k",
        "f1_at_valid_threshold",
        "ece",
        "latency_ms_per_1k",
    ]
    rows = []
    fit_time = {}
    for m in models:
        cells = [
            fmt(stat(by_seed(runs, m, k, fraud_rate=0.005)), 1 if k.startswith("lat") else 3)
            for k in sec
        ]
        fit = [
            r["fit_seconds"]
            for r in runs
            if r["model"] == m and r["condition"]["fraud_rate"] == 0.005
        ]
        fit_time[m] = stat(fit)
        rows.append([label(m), *cells, fmt(fit_time[m], 1)])
    write_table(
        "rq1_secondary",
        md_table(
            [
                "Model (0.5% fraud)",
                "ROC-AUC",
                "Precision@k",
                "Recall@k",
                "F1 @ valid thr.",
                "ECE",
                "ms / 1k preds",
                "Fit s",
            ],
            rows,
        ),
    )

    # Pre-registered comparisons, Holm within RQ1.
    comps = []
    for rate in RATES:
        base = by_seed(runs, "lgbm_tuned", "auprc", fraud_rate=rate)
        for m in MODERN:
            other = by_seed(runs, m, "auprc", fraud_rate=rate)
            if other:
                a, b = paired_arrays(other, base)
                c = paired(m, a, "lgbm_tuned", b)
                c.rate = rate  # type: ignore[attr-defined]
                comps.append(c)
    a, b = paired_arrays(
        by_seed(runs, "lgbm_tuned", "auprc", fraud_rate=0.001),
        by_seed(runs, "lgbm_default", "auprc", fraud_rate=0.001),
    )
    h1b = paired("lgbm_tuned", a, "lgbm_default", b)
    h1b.rate = 0.001  # type: ignore[attr-defined]
    comps = holm([*comps, h1b])
    rows = [
        [
            f"{c.rate:.1%}",
            label(c.a),
            label(c.b),
            f"{c.mean_diff:+.3f}",
            f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}]",
            f"{c.p_holm:.3f}",
            pretty_verdict(c.verdict, c.a, c.b),
        ]
        for c in comps
    ]
    write_table(
        "rq1_tests",
        md_table(
            ["Fraud rate", "A", "B", "Mean Δ AUPRC (A−B)", "95% CI", "Holm p", "Verdict"], rows
        ),
    )

    modern_beats = [c for c in comps if c.a in MODERN and c.verdict == f"{c.a} beats {c.b}"]
    h1a = "refuted" if modern_beats else "not refuted"
    h1b_v = "not refuted" if h1b.verdict == "lgbm_tuned beats lgbm_default" else "refuted"

    # Which model is best on mean AUPRC at each rate?
    best = {r: max(models, key=lambda m: auprc[m][r]["mean"]) for r in RATES}

    # Figure: one panel per fraud rate, the same AUPRC axis in each so the panels compare. The
    # unsupervised references sit below a rule.
    unsupervised = [m for m in models if m in UNSUPERVISED]
    ordered = [m for m in models if m not in UNSUPERVISED] + unsupervised
    top = max(auprc[m][r]["ci_high"] for m in models for r in RATES)
    xmax = math.ceil(top * 10) / 10
    fig, axes = figure(width=1.0, height=2.6, ncols=3, sharey=True)
    for panel, (ax, r) in zip("abc", zip(axes, RATES, strict=True), strict=True):
        dot_plot(
            ax,
            [(m, auprc[m][r]) for m in ordered],
            "Test AUPRC",
            separators=(len(ordered) - len(unsupervised),) if unsupervised else (),
        )
        ax.set_xlim(0, xmax)
        ax.set_xticks(np.arange(0, xmax + 1e-9, 0.2))
        ax.set_title(f"({panel}) {r:.1%} fraud")
    if unsupervised:
        # Name the group below the rule, in the panel whose right-hand side is empty there.
        axes[-1].text(
            xmax,
            len(unsupervised) - 0.5 - 0.08,
            "unsupervised references",
            ha="right",
            va="top",
            fontsize=7,
            color=INK_2,
            style="italic",
        )
    save(fig, "rq1_auprc")

    skipped = sorted({m for m in MODERN if m not in models})
    write_summary(
        "rq1",
        {
            "auprc": {m: {f"{r}": auprc[m][r] for r in RATES} for m in models},
            "fit_seconds_0p5": fit_time,
            "secondary_0p5": {
                m: {k: stat(by_seed(runs, m, k, fraud_rate=0.005)) for k in sec} for m in models
            },
            "tests": [c.to_dict() | {"rate": c.rate} for c in comps],
            "tests_keyed": {f"{c.a}_vs_{c.b}@{c.rate}": c.to_dict() for c in comps},
            "best_by_rate": {f"{r}": best[r] for r in RATES},
            "hypotheses": {"H1a": h1a, "H1b": h1b_v},
            "skipped_models": skipped,
            "n_seeds": max(auprc["lgbm_tuned"][r]["n"] for r in RATES),
        },
    )
    print("RQ1", {"H1a": h1a, "H1b": h1b_v, "best": best, "skipped": skipped})


if __name__ == "__main__":
    main()
