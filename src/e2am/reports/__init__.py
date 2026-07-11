"""Report generation: HTML, Markdown, PDF, and the cross-run leaderboard."""

from __future__ import annotations

from e2am.reports.generate import generate_run_artifacts
from e2am.reports.html import generate_html_report
from e2am.reports.leaderboard import update_leaderboard
from e2am.reports.markdown import generate_markdown_report
from e2am.reports.pdf import generate_pdf_report

__all__ = [
    "generate_html_report",
    "generate_markdown_report",
    "generate_pdf_report",
    "generate_run_artifacts",
    "update_leaderboard",
]
