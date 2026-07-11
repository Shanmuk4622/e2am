"""TensorBoard plugin: write per-epoch scalars to an event file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from e2am.plugins.base import Callback, require
from e2am.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from e2am.trainer.trainer import Trainer

logger = get_logger("plugins.tensorboard")


class TensorBoardPlugin(Callback):
    """Write every epoch metric as a TensorBoard scalar.

    Args:
        log_dir: Event-file directory. Defaults to
            ``<results>/<run_name>/tensorboard``.

    Raises:
        PluginError: If the ``tensorboard`` package is not installed.
    """

    def __init__(self, log_dir: str | Path | None = None) -> None:
        self._tb = require("torch.utils.tensorboard", pip_name="tensorboard")
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._writer: Any = None

    def on_fit_start(self, trainer: Trainer) -> None:
        log_dir = self.log_dir or trainer.config.output.dir / trainer.run_name / "tensorboard"
        self._writer = self._tb.SummaryWriter(log_dir=str(log_dir))
        logger.info("TensorBoard events -> %s", log_dir)

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        if self._writer is None:
            return
        for name, value in logs.items():
            self._writer.add_scalar(name, value, epoch)

    def on_fit_end(self, trainer: Trainer) -> None:
        if self._writer is None:
            return
        try:
            if trainer.result is not None:
                for key, value in trainer.result.to_flat_dict().items():
                    if isinstance(value, (int, float)):
                        self._writer.add_scalar(f"final/{key}", value)
        finally:
            self._writer.close()
            self._writer = None

    def on_exception(self, trainer: Trainer, exc: BaseException) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
