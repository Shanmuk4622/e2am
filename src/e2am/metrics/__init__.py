"""Model quality and Green AI metrics.

Green metrics and the tracker are dependency-light; the classification
metrics need PyTorch and are loaded lazily so ``import e2am.metrics`` works
on torch-free machines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from e2am.metrics.green import (
    DEFAULT_REFERENCE_KWH,
    GreenMetrics,
    compute_green_metrics,
    energy_accuracy_gradient,
    green_score,
)
from e2am.metrics.tracker import MetricsTracker

if TYPE_CHECKING:  # pragma: no cover
    from e2am.metrics.classification import (
        ClassificationMetrics,
        ClassMetrics,
        classification_metrics,
        confusion_matrix,
    )

__all__ = [
    "DEFAULT_REFERENCE_KWH",
    "ClassMetrics",
    "ClassificationMetrics",
    "GreenMetrics",
    "MetricsTracker",
    "classification_metrics",
    "compute_green_metrics",
    "confusion_matrix",
    "energy_accuracy_gradient",
    "green_score",
]

_TORCH_BACKED = {
    "ClassificationMetrics",
    "ClassMetrics",
    "classification_metrics",
    "confusion_matrix",
}


def __getattr__(name: str) -> Any:
    """Lazily import torch-backed classification metrics."""
    if name in _TORCH_BACKED:
        from e2am.metrics import classification

        return getattr(classification, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
