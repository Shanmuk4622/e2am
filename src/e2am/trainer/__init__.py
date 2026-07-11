"""Drop-in training loop with automatic Green AI telemetry.

The :class:`Trainer` itself requires PyTorch and is imported lazily; the
callback classes and :class:`TrainingResult` are torch-free, so reports and
plots for saved runs work on machines without PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from e2am.trainer.callbacks import Callback, EarlyStopping, LoggingCallback
from e2am.trainer.result import TrainingResult

if TYPE_CHECKING:  # pragma: no cover
    from e2am.trainer.trainer import Trainer

__all__ = [
    "Callback",
    "EarlyStopping",
    "LoggingCallback",
    "Trainer",
    "TrainingResult",
]


def __getattr__(name: str) -> Any:
    """Lazily import the torch-backed Trainer."""
    if name == "Trainer":
        from e2am.trainer.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
