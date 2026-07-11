"""One-call artifact generation for a finished run.

:func:`generate_run_artifacts` is the single entry point the Trainer, the
``monitor()`` context manager, and the CLI all use: given a result and an
output policy it writes plots, reports, and the leaderboard row — each step
individually guarded so one failing artifact never blocks the others.
"""

from __future__ import annotations

from pathlib import Path

from e2am.config.settings import OutputConfig
from e2am.reports.common import AnyResult
from e2am.utils.logging import get_logger

logger = get_logger("reports.generate")


def generate_run_artifacts(
    result: AnyResult,
    run_dir: str | Path,
    output: OutputConfig | None = None,
) -> list[Path]:
    """Write plots + reports + leaderboard for a run, per the output policy.

    Args:
        result: A finished training or monitoring result.
        run_dir: The run's artifact directory (``results/<run_name>``).
        output: Output policy; defaults to :class:`OutputConfig` defaults.

    Returns:
        Paths of every artifact successfully written.
    """
    run_dir = Path(run_dir)
    output = output or OutputConfig()
    written: list[Path] = []
    plots: list[Path] = []

    if output.save_plots:
        try:
            from e2am.visualization.plots import generate_plots

            plots = generate_plots(result, run_dir)
            written.extend(plots)
        except Exception as exc:
            logger.warning("Plot generation failed: %s", exc)

    try:
        from e2am.reports.markdown import generate_markdown_report

        written.append(generate_markdown_report(result, run_dir, plots or None))
    except Exception as exc:
        logger.warning("Markdown report failed: %s", exc)

    if output.save_html:
        try:
            from e2am.reports.html import generate_html_report

            written.append(generate_html_report(result, run_dir, plots or None))
        except Exception as exc:
            logger.warning("HTML report failed: %s", exc)

    if output.save_pdf:
        try:
            from e2am.reports.pdf import generate_pdf_report

            written.append(generate_pdf_report(result, run_dir, plots or None))
        except Exception as exc:
            logger.warning("PDF report failed: %s", exc)

    if output.save_csv:
        try:
            from e2am.reports.leaderboard import update_leaderboard

            written.append(update_leaderboard(result.to_flat_dict(), run_dir.parent))
        except Exception as exc:
            logger.warning("Leaderboard update failed: %s", exc)

    if written:
        logger.info("Wrote %d artifact(s) to %s", len(written), run_dir)
    return written
