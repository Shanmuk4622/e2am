"""Weights & Biases plugin: stream E2AM telemetry into a W&B run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from e2am.plugins.base import Callback, require
from e2am.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from e2am.trainer.trainer import Trainer

logger = get_logger("plugins.wandb")


class WandbPlugin(Callback):
    """Log every epoch's metrics — and the final green summary — to W&B.

    Args:
        project: W&B project (defaults to the E2AM project name).
        run_name: W&B run name (defaults to the E2AM run name).
        entity: W&B entity/team.
        **init_kwargs: Extra keyword arguments passed to ``wandb.init``.

    Raises:
        PluginError: If ``wandb`` is not installed.
    """

    def __init__(
        self,
        project: str | None = None,
        run_name: str | None = None,
        entity: str | None = None,
        **init_kwargs: Any,
    ) -> None:
        self._wandb = require("wandb")
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self.init_kwargs = init_kwargs
        self._run: Any = None

    def on_fit_start(self, trainer: Trainer) -> None:
        self._run = self._wandb.init(
            project=self.project or trainer.config.project,
            name=self.run_name or trainer.run_name,
            entity=self.entity,
            config=trainer.config.model_dump(mode="json"),
            **self.init_kwargs,
        )

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        if self._run is not None:
            self._run.log(dict(logs), step=epoch)

    def on_fit_end(self, trainer: Trainer) -> None:
        if self._run is None:
            return
        try:
            if trainer.result is not None:
                for key, value in trainer.result.to_flat_dict().items():
                    if isinstance(value, (int, float)):
                        self._run.summary[key] = value
        finally:
            self._run.finish()
            self._run = None

    def on_exception(self, trainer: Trainer, exc: BaseException) -> None:
        if self._run is not None:
            logger.warning("Training failed; closing W&B run with error status.")
            self._run.finish(exit_code=1)
            self._run = None
