"""PDF report generation (optional ``e2am[pdf]`` extra, via reportlab).

Renders the same sections as the HTML/markdown reports plus the plot images,
one summary document per run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from e2am.exceptions import ReportError
from e2am.reports.common import AnyResult, build_sections
from e2am.utils.logging import get_logger

logger = get_logger("reports.pdf")


def generate_pdf_report(
    result: AnyResult,
    run_dir: str | Path,
    plots: list[Path] | None = None,
) -> Path:
    """Write ``report.pdf`` into ``run_dir`` and return its path.

    Raises:
        ReportError: If reportlab is not installed
            (``pip install "e2am[pdf]"``).
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ReportError('PDF reports need reportlab: pip install "e2am[pdf]"') from exc

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if plots is None:
        plots = sorted(run_dir.glob("*.png"))
    path = run_dir / "report.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "E2AMTitle", parent=styles["Title"], fontSize=18, spaceAfter=4, alignment=0
    )
    subtitle_style = ParagraphStyle(
        "E2AMSubtitle", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#52514e")
    )
    heading_style = ParagraphStyle(
        "E2AMHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=6
    )

    table_style = TableStyle(
        [
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#52514e")),
            ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0b0b0b")),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e1e0d9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]
    )

    story: list = [
        Paragraph(f"{result.project} / {result.run_name}", title_style),
        Paragraph(
            f"E2AM experiment report · status {result.status} · generated "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            subtitle_style,
        ),
    ]
    for section_title, rows in build_sections(result):
        story.append(Paragraph(section_title, heading_style))
        body_style = ParagraphStyle("E2AMCell", parent=styles["Normal"], fontSize=8.5)
        data = [
            [Paragraph(label, body_style), Paragraph(value, body_style)] for label, value in rows
        ]
        table = Table(data, colWidths=[6 * cm, 10.5 * cm])
        table.setStyle(table_style)
        story.append(table)

    existing_plots = [plot for plot in plots if plot.exists()]
    if existing_plots:
        story.append(Paragraph("Charts", heading_style))
        for plot in existing_plots:
            story.append(Image(str(plot), width=16.5 * cm, height=9.28 * cm))
            story.append(Spacer(1, 0.4 * cm))

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=f"{result.project} / {result.run_name} · E2AM report",
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    document.build(story)
    logger.debug("Wrote PDF report to %s", path)
    return path
