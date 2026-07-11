"""Optimization engine: measured-data-driven efficiency suggestions."""

from __future__ import annotations

from e2am.optimize.analyzer import RULES, analyze
from e2am.optimize.suggestions import Suggestion

__all__ = ["RULES", "Suggestion", "analyze"]
