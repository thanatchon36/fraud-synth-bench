"""Render paper.tmpl.md and docs/EXEC_SUMMARY.tmpl.md from results/summary.json; build the PDF
and the self-contained web reader docs/paper/index.html (served by GitHub Pages).

Placeholders (every number in the paper comes from one of these):

* ``{{ rq1.auprc.lgbm_tuned.0.005.mean | .3f }}``: a value from summary.json by dotted
  path, then a format. Formats: any Python format spec, ``pct`` / ``pct1`` (percent),
  ``pm`` (mean ± std of a stat dict), ``ci`` (a stat dict's 95% CI), ``int``, ``str``.
* ``{{ table:rq1_auprc }}``: the Markdown table results/tables/rq1_auprc.md.
* ``{{ figure:rq1_auprc }}``: the path of results/figures/rq1_auprc.png.

A missing key is an error, never a blank. Usage: ``python paper/build.py [--no-pdf] [--no-html]``.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
TOKEN = re.compile(r"\{\{\s*(.+?)\s*\}\}")
READER = ROOT / "docs" / "paper" / "index.html"
REPO_URL = "https://github.com/thanatchon36/fraud-synth-bench"
DOI = "10.5281/zenodo.23343483"
DISCLAIMER = (
    "All results are on synthetic data. The generator encodes published patterns of fraud, not "
    "measurements of any real payment system; the results describe how models behave under "
    "controlled mechanisms, not production performance."
)
HYPOTHESIS = re.compile(r"^- \*\*(H\d[ab])(?: \([^)]*\))?:\*\* (.+)$", re.M)


def lookup(data: dict, path: str):
    """Dotted lookup that tolerates dots inside keys (e.g. ``0.005``)."""
    node, parts = data, path.split(".")
    i = 0
    while i < len(parts):
        for j in range(len(parts), i, -1):  # longest key first
            key = ".".join(parts[i:j])
            if isinstance(node, dict) and key in node:
                node, i = node[key], j
                break
        else:
            raise KeyError(f"summary.json has no key {'.'.join(parts[: i + 1])!r} (in {path!r})")
    return node


def fmt(value, spec: str) -> str:
    if value is None:
        return "n/a"
    if spec == "pm":
        return f"{value['mean']:.3f} ± {value['std']:.3f}"
    if spec == "ci":
        return f"[{value['ci_low']:.3f}, {value['ci_high']:.3f}]"
    if spec in ("pct", "pct0"):
        return f"{value * 100:.0f}%"
    if spec == "pct1":
        return f"{value * 100:.1f}%"
    if spec == "int":
        return f"{round(value):,}"
    if spec == "str":
        return str(value)
    return format(value, spec)


def render(text: str, summary: dict) -> str:
    def sub(m: re.Match) -> str:
        expr = m.group(1).replace("\\|", "|")  # "\|" escapes the pipe inside Markdown tables
        if expr.startswith("table:"):
            return (RESULTS / "tables" / f"{expr[6:].strip()}.md").read_text().strip()
        if expr.startswith("figure:"):
            return f"/results/figures/{expr[7:].strip()}.png"
        path, _, spec = (s.strip() for s in expr.partition("|"))
        return fmt(lookup(summary, path), spec or "str")

    out = TOKEN.sub(sub, text)
    left = TOKEN.findall(out)
    if left:
        raise ValueError(f"unrendered placeholders: {left}")
    return out


def build_pdf(md: Path) -> Path:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found: install pandoc >= 3.1 to build the PDF")
    import typst

    typ = PAPER / "build" / "paper.typ"
    typ.parent.mkdir(exist_ok=True)
    subprocess.run(
        [
            "pandoc",
            str(md),
            "--citeproc",
            "--bibliography",
            str(PAPER / "refs.bib"),
            "--csl",
            str(PAPER / "ieee.csl"),
            "-s",
            "-t",
            "typst",
            "-V",
            "mainfont=Libertinus Serif",
            "--resource-path",
            str(PAPER),
            "-o",
            str(typ),
        ],
        check=True,
        cwd=PAPER,
    )
    pdf = PAPER / "paper.pdf"
    typst.compile(str(typ), output=str(pdf), root=str(ROOT))
    return pdf


def _verdict_items(summary: dict) -> str:
    """The verdict strip: one item per hypothesis, its text from RESEARCH_QUESTIONS.md."""
    text = dict(HYPOTHESIS.findall((ROOT / "docs" / "RESEARCH_QUESTIONS.md").read_text()))
    items = []
    for hid, verdict in summary["hypotheses"].items():
        kept = verdict == "not refuted"
        kind = "kept" if kept else "refuted"
        statement = re.sub(r"\*\*(.+?)\*\*", r"\1", text.get(hid, ""))
        items.append(
            f'<li class="v v-{kind}"><span class="v-id">{hid}</span>'
            f'<span class="v-text">{html.escape(statement)}</span>'
            f'<span class="v-tag">{html.escape(verdict)}</span></li>'
        )
    return "\n".join(items)


def _meta_line(summary: dict) -> str:
    small = summary["data"]["small"]
    verdicts = list(summary["hypotheses"].values())
    held = sum(v == "not refuted" for v in verdicts)
    return " · ".join(
        [
            f"fraudlab v{summary['version']}",
            f"{summary['data']['n_scenarios']} fraud scenarios · "
            f"{small['n_transactions']:,} transactions per dataset · 5 paired seeds",
            f"<b>{len(verdicts)}</b> pre-registered hypotheses, <b>{held}</b> not refuted, "
            f"<b>{len(verdicts) - held}</b> refuted",
            f"suite runs in {summary['timing_minutes']:.0f} min on an "
            f"{summary['hardware']['cpu_brand']}",
        ]
    )


def _actions() -> str:
    links = [
        (f"{REPO_URL}/raw/main/paper/paper.pdf", "Download PDF"),
        (f"{REPO_URL}/blob/main/paper/paper.md", "Markdown source"),
        (f"{REPO_URL}/blob/main/results/summary.json", "results/summary.json"),
        (f"https://doi.org/{DOI}", f"DOI {DOI}"),
    ]
    return "".join(f'<li><a href="{href}">{html.escape(text)}</a></li>' for href, text in links)


def _colophon(summary: dict) -> str:
    hw = summary["hardware"]
    return (
        "<p>Every number, table and figure on this page is injected from "
        f'<a href="{REPO_URL}/blob/main/results/summary.json">results/summary.json</a> by '
        "<code>paper/build.py</code>; nothing is typed by hand. The PDF is typeset with Typst "
        "from the same Markdown. <code>make reproduce</code> rebuilds every result from the "
        "committed sources on a laptop.</p>"
        f"<p>{html.escape(DISCLAIMER)}</p>"
        f"<p>fraudlab {summary['version']} · Python {hw['python']} · PyTorch {hw['torch']} on "
        f"{hw['platform']} · Apache-2.0 · "
        f'<a href="{REPO_URL}">{REPO_URL.removeprefix("https://")}</a></p>'
    )


def build_html(md: Path, summary: dict) -> Path:
    """Convert the rendered paper to the self-contained web reader (figures embedded)."""
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found: install pandoc >= 3.1 to build the web reader")
    # Figure paths are root-absolute for Typst; make them relative for pandoc's resource lookup.
    source = md.read_text().replace("](/results/", "](results/")
    rendered = subprocess.run(
        [
            "pandoc",
            "--from=markdown+smart",
            "--to=html5",
            "--standalone",
            f"--template={PAPER / 'reader.html'}",
            "--toc",
            "--toc-depth=2",
            "--section-divs",
            "--number-sections",
            "--embed-resources",
            "--mathml",
            "--citeproc",
            f"--bibliography={PAPER / 'refs.bib'}",
            f"--csl={PAPER / 'ieee.csl'}",
            f"--resource-path={ROOT}",
        ],
        input=source,
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    fills = {
        "<!--KICKER-->": "Reproducible study · Fraud detection on synthetic data · "
        f"{len(summary['hypotheses'])} pre-registered hypotheses",
        "<!--META-->": _meta_line(summary),
        "<!--ACTIONS-->": _actions(),
        "<!--DISCLAIMER-->": f"<b>Limitations first.</b> {html.escape(DISCLAIMER)}",
        "<!--VERDICT-NOTE-->": "Hypotheses, metrics and thresholds were fixed in "
        "<code>docs/RESEARCH_QUESTIONS.md</code> before any data was generated. A verdict is "
        "mechanical against the recorded threshold; an inconclusive result counts as a refutation.",
        "<!--VERDICTS-->": _verdict_items(summary),
        "<!--COLOPHON-->": _colophon(summary),
    }
    for token, value in fills.items():
        if token not in rendered:
            raise RuntimeError(f"reader.html no longer contains {token}")
        rendered = rendered.replace(token, value)
    READER.parent.mkdir(parents=True, exist_ok=True)
    READER.write_text(rendered)
    return READER


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    md = PAPER / "paper.md"
    md.write_text(render((PAPER / "paper.tmpl.md").read_text(), summary))
    print(f"wrote {md.relative_to(ROOT)}")
    exec_tmpl = ROOT / "docs" / "EXEC_SUMMARY.tmpl.md"
    if exec_tmpl.exists():
        out = ROOT / "docs" / "EXEC_SUMMARY.md"
        out.write_text(render(exec_tmpl.read_text(), summary))
        print(f"wrote {out.relative_to(ROOT)}")
    if "--no-pdf" not in sys.argv:
        print(f"wrote {build_pdf(md).relative_to(ROOT)}")
    if "--no-html" not in sys.argv:
        print(f"wrote {build_html(md, summary).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
