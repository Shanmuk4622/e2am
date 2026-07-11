"""Hugging Face ``transformers`` integration.

Add one callback to any HF ``Trainer`` and E2AM records energy, carbon,
utilization, and green metrics for the run — and writes the standard E2AM
artifact set (metrics.json, plots, reports, leaderboard row) next to your
other results:

    >>> from transformers import Trainer
    >>> from e2am.integrations import E2AMCallback
    >>> trainer = Trainer(model=model, args=args, train_dataset=ds,
    ...                   callbacks=[E2AMCallback(project="bert-finetune")])
    >>> trainer.train()

Requires ``transformers`` (``pip install transformers``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError as exc:  # pragma: no cover - depends on environment
    raise ImportError(
        "e2am.integrations.huggingface requires the 'transformers' package: "
        "pip install transformers"
    ) from exc

from e2am.config.settings import ExperimentConfig, MonitorConfig
from e2am.metrics.green import compute_green_metrics
from e2am.metrics.tracker import MetricsTracker
from e2am.monitoring.session import MonitorSession
from e2am.trainer.result import TrainingResult
from e2am.utils.logging import get_logger

logger = get_logger("integrations.huggingface")


class E2AMCallback(TrainerCallback):
    """Green AI telemetry for the Hugging Face ``Trainer``.

    Args:
        project: E2AM project name grouping related runs.
        run_name: Run identifier; auto-generated when omitted.
        output_dir: E2AM results root (default ``results``).
        sampling_interval_s: Hardware sampling interval.
        eval_accuracy_key: Metric name in HF logs used as "accuracy" for
            green metrics (default ``"eval_accuracy"``).
        save_artifacts: Write the E2AM artifact set when training ends.
    """

    def __init__(
        self,
        project: str = "e2am",
        run_name: str | None = None,
        output_dir: str | Path | None = None,
        sampling_interval_s: float = 1.0,
        eval_accuracy_key: str = "eval_accuracy",
        save_artifacts: bool = True,
    ) -> None:
        self.config = ExperimentConfig(
            project=project,
            run_name=run_name,
            monitor=MonitorConfig(sampling_interval_s=sampling_interval_s),
        )
        if output_dir is not None:
            self.config.output.dir = Path(output_dir)
        self.run_name = self.config.resolved_run_name()
        self.config.run_name = self.run_name
        self.eval_accuracy_key = eval_accuracy_key
        self.save_artifacts = save_artifacts

        self.tracker = MetricsTracker()
        self.result: TrainingResult | None = None
        self._session: MonitorSession | None = None
        self._epoch_accuracies: list[float] = []
        self._epoch_energy_wh: list[float] = []

    # ------------------------------------------------------------------
    # HF Trainer lifecycle
    # ------------------------------------------------------------------

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._session = MonitorSession(
            config=self.config.monitor,
            project=self.config.project,
            run_name=self.run_name,
        )
        self._session.start()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        numeric = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        if numeric:
            self.tracker.log(numeric, step=int(getattr(state, "global_step", 0) or 0))
        if self.eval_accuracy_key in numeric and self._session is not None:
            snapshot = self._session.snapshot()
            self._epoch_accuracies.append(numeric[self.eval_accuracy_key])
            self._epoch_energy_wh.append(float(snapshot["total_energy_j"]) / 3600.0)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._session is not None:
            snapshot = self._session.snapshot()
            self.tracker.log(
                {"cumulative_energy_wh": float(snapshot["total_energy_j"]) / 3600.0},
                step=int(getattr(state, "global_step", 0) or 0),
            )

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._session is None:
            return
        monitor_result = self._session.stop()
        self._session = None

        epochs = int(getattr(state, "epoch", 0) or 0)
        global_step = int(getattr(state, "global_step", 0) or 0)
        batch = int(getattr(args, "per_device_train_batch_size", 0) or 0)
        accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        samples = global_step * batch * accum  # approximate (single process)

        best_acc = max(self._epoch_accuracies) if self._epoch_accuracies else None
        green = compute_green_metrics(
            energy_j=monitor_result.total_energy_j,
            num_samples=samples or None,
            accuracy=best_acc,
            emissions_g=monitor_result.carbon.emissions_g,
            epoch_accuracies=self._epoch_accuracies or None,
            epoch_cumulative_energy_wh=self._epoch_energy_wh or None,
        )

        mixed_precision = bool(getattr(args, "fp16", False) or getattr(args, "bf16", False))
        device = "cuda" if monitor_result.system.cuda_available else "cpu"
        self.result = TrainingResult(
            project=self.config.project,
            run_name=self.run_name,
            epochs_requested=int(getattr(args, "num_train_epochs", epochs) or epochs),
            epochs_completed=epochs,
            device=device,
            mixed_precision=mixed_precision,
            samples_processed=samples,
            best_val_accuracy=best_acc,
            final_val_accuracy=(self._epoch_accuracies[-1] if self._epoch_accuracies else None),
            final_train_loss=self.tracker.latest("loss"),
            final_val_loss=self.tracker.latest("eval_loss"),
            monitor=monitor_result,
            green=green,
            history=self.tracker.to_dict(),
        )
        if self.save_artifacts:
            run_dir = self.config.output.dir / self.run_name
            try:
                self.result.save(run_dir)
                self.config.to_yaml(run_dir / "config.yaml")
                from e2am.reports.generate import generate_run_artifacts

                generate_run_artifacts(self.result, run_dir, self.config.output)
                logger.info("E2AM artifacts for HF run saved to %s", run_dir)
            except Exception as exc:
                logger.warning("Could not save E2AM artifacts: %s", exc)
