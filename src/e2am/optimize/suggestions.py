"""Suggestion model produced by the optimization analyzer."""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Display order for priorities.
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class Suggestion(BaseModel):
    """One actionable optimization with its estimated impact."""

    id: str = Field(description="Stable machine-readable identifier (kebab-case).")
    title: str
    priority: str = Field(default="medium", pattern="^(high|medium|low)$")
    rationale: str = Field(description="Why this fired, grounded in the run's measurements.")
    action: str = Field(description="Concrete change, usually a code snippet.")
    estimated_savings_pct: tuple[float, float] | None = Field(
        default=None,
        description="Rough (low, high) percentage of run energy this could save.",
    )
    estimated_savings_wh: float | None = Field(
        default=None,
        description="Quantified saving in Wh when derivable from the run itself.",
    )

    def savings_label(self) -> str:
        """Human-readable savings estimate."""
        if self.estimated_savings_wh is not None:
            return f"~{self.estimated_savings_wh:.4g} Wh (measured from this run)"
        if self.estimated_savings_pct is not None:
            low, high = self.estimated_savings_pct
            return f"~{low:.0f}–{high:.0f} % (typical range)"
        return "—"
