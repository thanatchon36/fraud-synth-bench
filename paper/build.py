"""Render paper.tmpl.md and docs/EXEC_SUMMARY.tmpl.md from results/summary.json; build the PDF.

Placeholders (every number in the paper comes from one of these):

* ``{{ rq1.auprc.lgbm_tuned.0.005.mean | .3f }}``: a value from summary.json by dotted
  path, then a format. Formats: any Python format spec, ``pct`` / ``pct1`` (percent),
  ``pm`` (mean ± std of a stat dict), ``ci`` (a stat dict's 95% CI), ``int``, ``str``.
* ``{{ table:rq1_auprc }}``: the Markdown table results/tables/rq1_auprc.md.
* ``{{ figure:rq1_auprc }}``: the path of results/figures/rq1_auprc.png.

A missing key is an error, never a blank. Usage: ``python paper/build.py [--no-pdf]``.
"""

from __future__ import annotations

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
        expr = m.group(1)
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


if __name__ == "__main__":
    main()
