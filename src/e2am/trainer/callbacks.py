"""Callback lifecycle for the E2AM Trainer.

Callbacks are the single extension mechanism of the training loop — progress
bars, early stopping, and every future integration plugin (W&B, MLflow,
Slack, ...) implement this same interface. Hooks receive the live
:class:`~e2am.trainer.trainer.Trainer`, so they can read
``trainer.tracker``, ``trainer.monitor_session``, or request a stop by
setting ``trainer.should_stop = True``.

Hook failures are logged and swallowed by the trainer: a broken progress bar
must never kill an eight-hour run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from e2am.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from e2am.trainer.trainer import Trainer

logger = get_logger("trainer.callbacks")


class Callback:
    """Base class with no-op hooks; override only what you need."""

    def on_fit_start(self, trainer: Trainer) -> None:
        """Called once before the first epoch."""

    def on_fit_end(self, trainer: Trainer) -> None:
        """Called once after training finishes (also after failures)."""

    def on_epoch_start(self, trainer: Trainer, epoch: int) -> None:
        """Called at the start of every epoch (0-based)."""

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        """Called after train + validation of an epoch, with its metrics."""

    def on_batch_start(self, trainer: Trainer, batch_idx: int) -> None:
        """Called before each training batch."""

    def on_batch_end(self, trainer: Trainer, batch_idx: int, logs: dict[str, float]) -> None:
        """Called after each training batch, with batch metrics."""

    def on_exception(self, trainer: Trainer, exc: BaseException) -> None:
        """Called when training aborts with an exception."""


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name from the epoch logs (e.g. ``"val_loss"``,
            ``"val_accuracy"``).
        patience: Epochs without improvement tolerated before stopping.
        mode: ``"min"`` if lower is better, ``"max"`` if higher is better.
        min_delta: Minimum change that counts as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.best: float = math.inf if mode == "min" else -math.inf
        self.wait = 0
        self.stopped_epoch: int | None = None

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        value = logs.get(self.monitor)
        if value is None:
            logger.warning(
                "EarlyStopping: metric %r not in epoch logs %s; skipping check.",
                self.monitor,
                sorted(logs),
            )
            return
        if self._improved(value):
            self.best = value
            self.wait = 0
            return
        self.wait += 1
        if self.wait > self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True
            logger.info(
                "EarlyStopping: %s did not improve for %d epochs; stopping at epoch %d.",
                self.monitor,
                self.wait,
                epoch,
            )


class LoggingCallback(Callback):
    """Log a one-line summary after every epoch."""

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        parts = [f"{name}={value:.4f}" for name, value in sorted(logs.items())]
        logger.info("Epoch %d/%d · %s", epoch + 1, trainer.epochs, " · ".join(parts))


class CallbackRunner:
    """Dispatches lifecycle events to callbacks, isolating their failures."""

    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks = callbacks

    def fire(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Invoke ``event`` on every callback; log-and-continue on errors."""
        for callback in self.callbacks:
            try:
                getattr(callback, event)(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Callback %s.%s failed: %s",
                    type(callback).__name__,
                    event,
                    exc,
                )
