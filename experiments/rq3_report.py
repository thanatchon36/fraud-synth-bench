"""RQ3 report: GraphSAGE vs. LightGBM with and without graph features."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    INK_2,
    by_seed,
    color,
    fmt,
    label,
    load_runs,
    md_table,
    paired_arrays,
    save,
    stat,
    style_axes,
    write_summary,
    write_table,
)
from fraudlab.eval import holm, paired
from fraudlab.synth import RING_SCENARIOS

SHARES = [0.0, 0.25, 0.5, 0.75]
MODELS = ["lgbm_tuned", "lgbm_tuned+graph", "graphsage"]


def ring_recall(runs: list[dict], model: str, share: float) -> dict[int, float]:
    out = {}
    for r in runs:
        if r["model"] == model and r["condition"]["ring_share"] == share:
            vals = [v for k, v in r["recall_at_k_by_scenario"].items() if k in RING_SCENARIOS]
            if vals:
                out[r["seed"]] = float(np.mean(vals))
    return out


def main() -> None:
    runs = load_runs("rq3")
    auprc = {m: {s: stat(by_seed(runs, m, "auprc", ring_share=s)) for s in SHARES} for m in MODELS}
    rrec = {m: {s: stat(ring_recall(runs, m, s)) for s in SHARES if s > 0} for m in MODELS}
    rows = [
        [f"{s:.0%}"]
        + [fmt(auprc[m][s]) for m in MODELS]
        + [fmt(rrec[m][s]) if s > 0 else "–" for m in MODELS]
        for s in SHARES
    ]
    write_table(
        "rq3_auprc",
        md_table(
            ["Ring share"]
            + [f"AUPRC: {label(m)}" for m in MODELS]
            + [f"Ring recall@k: {label(m)}" for m in MODELS],
            rows,
        ),
    )

    comps = []
    for s in SHARES:
        a, b = paired_arrays(
            by_seed(runs, "graphsage", "auprc", ring_share=s),
            by_seed(runs, "lgbm_tuned+graph", "auprc", ring_share=s),
        )
        c = paired("graphsage", a, "lgbm_tuned+graph", b)
        c.share, c.h = s, "H3a"  # type: ignore[attr-defined]
        comps.append(c)
    for s in (0.5, 0.75):
        a, b = paired_arrays(
            by_seed(runs, "lgbm_tuned+graph", "auprc", ring_share=s),
            by_seed(runs, "lgbm_tuned", "auprc", ring_share=s),
        )
        c = paired("lgbm_tuned+graph", a, "lgbm_tuned", b)
        c.share, c.h = s, "H3b"  # type: ignore[attr-defined]
        comps.append(c)
    comps = holm(comps)
    write_table(
        "rq3_tests",
        md_table(
            ["Hyp.", "Ring share", "A", "B", "Mean Δ AUPRC", "95% CI", "Holm p", "Verdict"],
            [
                [
                    c.h,
                    f"{c.share:.0%}",
                    label(c.a),
                    label(c.b),
                    f"{c.mean_diff:+.3f}",
                    f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}]",
                    f"{c.p_holm:.3f}",
                    c.verdict,
                ]
                for c in comps
            ],
        ),
    )

    h3a = (
        "refuted"
        if any(c.h == "H3a" and c.verdict.startswith("graphsage beats") for c in comps)
        else "not refuted"
    )
    h3b_wins = [c for c in comps if c.h == "H3b" and c.verdict.startswith("lgbm_tuned+graph beats")]
    h3b = "not refuted" if h3b_wins else "refuted"

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.array(SHARES) * 100
    for i, m in enumerate(MODELS):
        mean = np.array([auprc[m][s]["mean"] for s in SHARES])
        lo = np.array([auprc[m][s]["ci_low"] for s in SHARES])
        hi = np.array([auprc[m][s]["ci_high"] for s in SHARES])
        off = (i - 1) * 1.2
        ax.errorbar(
            x + off,
            mean,
            yerr=[mean - lo, hi - mean],
            color=color(m),
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=0,
            label=label(m),
        )
    ax.set_xticks(x, [f"{s:.0%}" for s in SHARES])
    ax.set_xlabel("Share of fraud from ring scenarios", color=INK_2)
    ax.set_ylabel("Test AUPRC (mean, 95% CI)", color=INK_2)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title(
        "RQ3: graph structure vs. model family (small preset, 0.5% fraud)", loc="left", fontsize=11
    )
    save(fig, "rq3_ring_share")

    write_summary(
        "rq3",
        {
            "auprc": {m: {f"{s}": auprc[m][s] for s in SHARES} for m in MODELS},
            "ring_recall": {m: {f"{s}": v for s, v in rrec[m].items()} for m in MODELS},
            "tests": [c.to_dict() | {"share": c.share, "hypothesis": c.h} for c in comps],
            "tests_keyed": {f"{c.a}_vs_{c.b}@{c.share}": c.to_dict() for c in comps},
            "hypotheses": {"H3a": h3a, "H3b": h3b},
        },
    )
    print("RQ3", {"H3a": h3a, "H3b": h3b})


if __name__ == "__main__":
    main()
