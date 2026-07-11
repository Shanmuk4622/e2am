"""Cross-run leaderboard maintenance.

Every finished run appends its headline numbers to ``leaderboard.csv`` at the
results root, so all experiments of a project stay comparable in one file.
Re-running with the same ``run_name`` replaces that run's row instead of
duplicating it.

Implemented with the stdlib ``csv`` module on purpose: the leaderboard is the
one artifact that must never fail, so it depends on nothing optional.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from e2am.utils.logging import get_logger

logger = get_logger("reports.leaderboard")


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        logger.warning("Could not read existing leaderboard (%s); rewriting it.", exc)
        return []


def update_leaderboard(flat_row: dict[str, Any], results_root: str | Path) -> Path:
    """Insert or replace one run's row in ``<results_root>/leaderboard.csv``.

    Args:
        flat_row: Headline metrics (from ``result.to_flat_dict()``); should
            contain a ``run_name`` key, which is the row identity.
        results_root: Directory holding all run directories.

    Returns:
        Path of the updated CSV.
    """
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / "leaderboard.csv"

    new_row = {key: "" if value is None else str(value) for key, value in flat_row.items()}
    rows = _read_rows(path) if path.exists() else []
    run_name = new_row.get("run_name")
    if run_name is not None:
        rows = [row for row in rows if row.get("run_name") != run_name]
    rows.append(new_row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    logger.debug("Leaderboard updated: %s (%d runs)", path, len(rows))
    return path
