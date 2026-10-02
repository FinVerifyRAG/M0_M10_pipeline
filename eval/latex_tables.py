"""Write LaTeX tables from one results object."""
from __future__ import annotations

from pathlib import Path


def _cell(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("_", "\\_")


def write_latex(results: dict, path: str | Path) -> None:
    summary = results.get("summary") or {}
    main = results.get("main_table") or {}
    rows = main.get("rows") if isinstance(main, dict) else None
    lines = [
        "% Generated from one results object. Do not edit by hand.",
        f"% config {summary.get('config_hash', '')}",
        "\\begin{tabular}{lrrrr}",
        "System & Hallucination & Coverage & AURC & Judge calls \\\\",
        "\\hline",
    ]
    if rows:
        for row in rows:
            lines.append(
                f"{_cell(row.get('system'))} & {_cell(row.get('hallucination_rate'))} "
                f"& {_cell(row.get('coverage'))} & {_cell(row.get('aurc'))} "
                f"& {_cell(row.get('judge_call_rate'))} \\\\"
            )
    else:
        lines.append(
            f"RegGuard & {_cell(summary.get('hallucination_rate'))} "
            f"& {_cell(summary.get('coverage'))} & {_cell(summary.get('aurc'))} & -- \\\\"
        )
    lines.append("\\end{tabular}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
