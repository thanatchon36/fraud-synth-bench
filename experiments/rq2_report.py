"""RQ2 report: concept drift. Inputs: results/runs/rq2.jsonl."""

from __future__ import annotations

import numpy as np

from _common import (
    INK_2,
    color,
    figure,
    fmt,
    label,
    legend_above,
    load_runs,
    md_table,
    save,
    series_handle,
    stat,
    style_axes,
    write_summary,
    write_table,
)

SUPERVISED = ["logreg", "lgbm_tuned", "gru"]
UNSUPERVISED = ["iforest", "autoencoder"]
DRIFT_DAY = 54


def _ci(s: dict) -> str:
    return f"[{s['ci_low']:.2f}, {s['ci_high']:.2f}]"


def seg(r: dict, name: str) -> float:
    v = r["windows"]["segments"][name]["auprc"]
    return float("nan") if v is None else v


def main() -> None:
    runs = load_runs("rq2")
    static = {m: {r["seed"]: r for r in runs if r["model"] == m} for m in SUPERVISED + UNSUPERVISED}
    # H2b policy: refit on all labels (D029). The holdout re-tune is reported alongside.
    retrain = {r["seed"]: r for r in runs if r["model"] == "lgbm_tuned+refit"}
    holdout = {r["seed"]: r for r in runs if r["model"] == "lgbm_tuned+retrain"}

    rows, deg = [], {}
    for m, by in static.items():
        pre = {s: seg(r, "pre") for s, r in by.items()}
        post = {s: seg(r, "post") for s, r in by.items()}
        d = {s: 1 - post[s] / pre[s] for s in by if pre[s] > 0}
        deg[m] = {"pre": stat(pre), "post": stat(post), "degradation": stat(d)}
        rows.append(
            [
                label(m),
                "supervised" if m in SUPERVISED else "unsupervised",
                fmt(deg[m]["pre"]),
                fmt(deg[m]["post"]),
                fmt(deg[m]["degradation"], 2),
                f"[{deg[m]['degradation']['ci_low']:.2f}, {deg[m]['degradation']['ci_high']:.2f}]",
            ]
        )
    write_table(
        "rq2_degradation",
        md_table(
            [
                "Model (static)",
                "Type",
                "AUPRC pre-drift (d50–54)",
                "AUPRC post-drift (d54–90)",
                "Degradation D",
                "95% CI of D",
            ],
            rows,
        ),
    )

    # H2a
    sup_refute = [m for m in SUPERVISED if deg[m]["degradation"]["ci_high"] < 0.30]
    uns_refute = [m for m in UNSUPERVISED if deg[m]["degradation"]["ci_low"] > 0.30]
    h2a = "refuted" if (sup_refute or uns_refute) else "not refuted"

    # H2b: recovery of LightGBM with weekly retraining over days 61–68.
    def recovery(policy_runs: dict) -> dict:
        out = {}
        for s_, r in policy_runs.items():
            st = static["lgbm_tuned"].get(s_)
            if st is None:
                continue
            a_pre, a_static, a_re = seg(st, "pre"), seg(st, "recovery"), seg(r, "recovery")
            gap = a_pre - a_static
            out[s_] = {
                "pre": a_pre,
                "static": a_static,
                "retrain": a_re,
                "R": (a_re - a_static) / gap if gap > 0 else float("nan"),
            }
        return out

    rec = recovery(retrain)
    rec_holdout = recovery(holdout)
    R = stat({s: v["R"] for s, v in rec.items()})
    R_holdout = stat({s: v["R"] for s, v in rec_holdout.items()})
    h2b = "refuted" if R["ci_high"] < 0.5 else "not refuted"
    write_table(
        "rq2_recovery",
        md_table(
            ["Quantity (LightGBM, days 61–68)", "Mean ± std"],
            [
                [
                    "AUPRC pre-drift (static, d50–54)",
                    fmt(stat({s: v["pre"] for s, v in rec.items()})),
                ],
                ["AUPRC static", fmt(stat({s: v["static"] for s, v in rec.items()}))],
                [
                    "AUPRC weekly refit on all labels (H2b)",
                    fmt(stat({s: v["retrain"] for s, v in rec.items()})),
                ],
                ["Recovery R, refit", f"{fmt(R, 2)} (95% CI {_ci(R)})"],
                [
                    "AUPRC weekly re-tune with 10-day holdout",
                    fmt(stat({s: v["retrain"] for s, v in rec_holdout.items()})),
                ],
                ["Recovery R, holdout re-tune", f"{fmt(R_holdout, 2)} (95% CI {_ci(R_holdout)})"],
            ],
        ),
    )

    # Figure: weekly AUPRC over time, one line per static model and the dashed refit policy.
    fig, ax = figure(width=1.0, height=2.9)
    series = [*SUPERVISED, *UNSUPERVISED, "lgbm_tuned+refit"]
    handles, n_seeds, starts = [], 0, []
    for m in series:
        rs = list(retrain.values()) if m.endswith("+refit") else list(static[m].values())
        if not rs:
            continue
        starts = [w["start"] for w in rs[0]["windows"]["weekly"]]
        vals = np.array(
            [
                [np.nan if w["auprc"] is None else w["auprc"] for w in r["windows"]["weekly"]]
                for r in rs
            ]
        )
        n_seeds = max(n_seeds, len(rs))
        dashed = m.endswith("+refit")
        ax.plot(
            starts,
            np.nanmean(vals, axis=0),
            marker="o",
            markersize=3.6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.1,
            color=color(m),
            linestyle=(0, (4, 2)) if dashed else "-",
            zorder=3,
        )
        handles.append(series_handle(m, dashed=dashed))
    ax.axvline(DRIFT_DAY, color=INK_2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.text(
        DRIFT_DAY + 0.4,
        0.985,
        f"fraud tactics switch (day {DRIFT_DAY})",
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK_2,
    )
    ax.set_xticks(starts)
    ax.set_xlim(min(starts) - 1.5, max(starts) + 1.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Start of the 7-day test window (day)")
    ax.set_ylabel(f"Test AUPRC (mean of {n_seeds} seeds)")
    style_axes(ax, grid="y")
    legend_above(ax, handles, ncols=3)
    save(fig, "rq2_weekly")

    write_summary(
        "rq2",
        {
            "degradation": deg,
            "recovery": {
                "R": R,
                "per_seed": rec,
                "R_holdout": R_holdout,
                "per_seed_holdout": rec_holdout,
            },
            "hypotheses": {"H2a": h2a, "H2b": h2b},
            "h2a_refuting_models": sup_refute + uns_refute,
        },
    )
    print("RQ2", {"H2a": h2a, "H2b": h2b, "refuting": sup_refute + uns_refute, "R": R["mean"]})


if __name__ == "__main__":
    main()
