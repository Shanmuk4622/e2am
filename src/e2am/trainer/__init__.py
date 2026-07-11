"""Drop-in training loop with automatic Green AI telemetry (requires PyTorch)."""

from __future__ import annotations

from e2am.trainer.callbacks import Callback, EarlyStopping, LoggingCallback
from e2am.trainer.result import TrainingResult
from e2am.trainer.trainer import Trainer

__all__ = [
    "Callback",
    "EarlyStopping",
    "LoggingCallback",
    "Trainer",
    "TrainingResult",
]
