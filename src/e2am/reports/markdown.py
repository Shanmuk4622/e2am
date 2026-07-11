"""Per-run ``README.md`` generation.

A run directory should explain itself when browsed on GitHub or in an
editor: the README carries the same sections as the HTML report, in
markdown tables, with relative links to the plot images.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import e2am
from e2am.reports.common import AnyResult, build_sections
from e2am.utils.logging import get_logger

logger = get_logger("reports.markdown")


def generate_markdown_report(
    result: AnyResult,
    run_dir: str | Path,
    plots: list[Path] | None = None,
) -> Path:
    """Write ``README.md`` into ``run_dir`` and return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if plots is None:
        plots = sorted(run_dir.glob("*.png"))

    lines: list[str] = [
        f"# {result.project} / {result.run_name}",
        "",
        f"Status: **{result.status}** · generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        f"[E2AM](https://github.com/Shanmuk4622/e2am) v{e2am.__version__}",
    ]
    for title, rows in build_sections(result):
        lines += ["", f"## {title}", "", "| | |", "|---|---|"]
        lines += [f"| {label} | {value} |" for label, value in rows]

    if plots:
        lines += ["", "## Charts", ""]
        lines += [f"![{plot.stem}]({plot.name})" for plot in plots if plot.exists()]

    path = run_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.debug("Wrote markdown report to %s", path)
    return path
