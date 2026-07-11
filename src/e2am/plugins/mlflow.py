"""MLflow plugin: track E2AM runs as MLflow runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from e2am.plugins.base import Callback, require
from e2am.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from e2am.trainer.trainer import Trainer

logger = get_logger("plugins.mlflow")


def _flatten(prefix: str, data: dict[str, Any], out: dict[str, str]) -> None:
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten(name, value, out)
        else:
            out[name] = str(value)


class MLflowPlugin(Callback):
    """Log parameters, per-epoch metrics, and the final summary to MLflow.

    Args:
        tracking_uri: MLflow tracking server URI (defaults to MLflow's own
            default, usually the local ``mlruns/`` directory).
        experiment_name: Experiment to log under (defaults to the E2AM
            project name).
        run_name: MLflow run name (defaults to the E2AM run name).

    Raises:
        PluginError: If ``mlflow`` is not installed.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
    ) -> None:
        self._mlflow = require("mlflow")
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._active = False

    def on_fit_start(self, trainer: Trainer) -> None:
        if self.tracking_uri:
            self._mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow.set_experiment(self.experiment_name or trainer.config.project)
        self._mlflow.start_run(run_name=self.run_name or trainer.run_name)
        self._active = True
        params: dict[str, str] = {}
        _flatten("", trainer.config.model_dump(mode="json"), params)
        self._mlflow.log_params(params)

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        if self._active:
            self._mlflow.log_metrics(dict(logs), step=epoch)

    def on_fit_end(self, trainer: Trainer) -> None:
        if not self._active:
            return
        try:
            if trainer.result is not None:
                final = {
                    key: float(value)
                    for key, value in trainer.result.to_flat_dict().items()
                    if isinstance(value, (int, float))
                }
                self._mlflow.log_metrics(final)
        finally:
            self._mlflow.end_run()
            self._active = False

    def on_exception(self, trainer: Trainer, exc: BaseException) -> None:
        if self._active:
            logger.warning("Training failed; ending MLflow run as FAILED.")
            self._mlflow.end_run(status="FAILED")
            self._active = False
