"""Result model produced by :meth:`e2am.Trainer.fit`.

Bundles everything one run produced — energy/carbon monitoring, the static
model profile, quality metrics, green metrics, and per-epoch history — into
one JSON-serializable object that reports are generated from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from e2am.metrics.green import GreenMetrics
from e2am.monitoring.result import MonitorResult


class TrainingResult(BaseModel):
    """Complete outcome of one training run."""

    project: str = "e2am"
    run_name: str = ""
    status: str = Field(default="completed", description="'completed', 'stopped', or 'failed'.")
    epochs_requested: int = 0
    epochs_completed: int = 0
    samples_processed: int = Field(
        default=0, description="Total training samples seen across all epochs."
    )
    best_val_accuracy: float | None = None
    best_epoch: int | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    final_val_accuracy: float | None = None
    final_val_f1_macro: float | None = None
    avg_batch_time_ms: float | None = None
    avg_epoch_time_s: float | None = None
    train_throughput_samples_per_s: float | None = None
    monitor: MonitorResult | None = None
    profile: dict[str, Any] | None = Field(
        default=None, description="ModelProfile dump (params, MACs, FLOPs, coverage)."
    )
    green: GreenMetrics | None = None
    history: dict[str, dict[str, list[float]]] = Field(
        default_factory=dict,
        description="Per-epoch series from the MetricsTracker "
        "({name: {'steps': [...], 'values': [...]}}).",
    )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def total_energy_wh(self) -> float | None:
        """Total measured energy in Wh, when monitoring ran."""
        return self.monitor.total_energy_wh if self.monitor else None

    @property
    def emissions_g(self) -> float | None:
        """Total carbon emissions in gCO2eq, when monitoring ran."""
        return self.monitor.carbon.emissions_g if self.monitor else None

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten headline numbers for leaderboards and CSV export."""
        flat: dict[str, Any] = {
            "project": self.project,
            "run_name": self.run_name,
            "status": self.status,
            "epochs_completed": self.epochs_completed,
            "samples_processed": self.samples_processed,
            "best_val_accuracy": self.best_val_accuracy,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "final_val_accuracy": self.final_val_accuracy,
            "avg_epoch_time_s": self.avg_epoch_time_s,
            "train_throughput_samples_per_s": self.train_throughput_samples_per_s,
        }
        if self.monitor is not None:
            monitor_flat = self.monitor.to_flat_dict()
            for key in ("project", "run_name", "status"):
                monitor_flat.pop(key, None)
            flat.update(monitor_flat)
        if self.profile is not None:
            flat["params"] = self.profile.get("params")
            flat["macs"] = self.profile.get("macs")
            flat["flops"] = self.profile.get("flops")
        if self.green is not None:
            flat["energy_per_sample_j"] = self.green.energy_per_sample_j
            flat["carbon_per_sample_mg"] = self.green.carbon_per_sample_mg
            flat["accuracy_per_kwh"] = self.green.accuracy_per_kwh
            flat["green_score"] = self.green.green_score
            flat["eag_pct_per_wh"] = self.green.eag_pct_per_wh
        return flat

    def save(self, run_dir: str | Path) -> Path:
        """Write ``metrics.json`` and a flat ``summary.yaml`` into ``run_dir``."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(self.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "summary.yaml").write_text(
            yaml.safe_dump(self.to_flat_dict(), sort_keys=False), encoding="utf-8"
        )
        return run_dir

    @classmethod
    def load(cls, run_dir: str | Path) -> TrainingResult:
        """Load a result previously saved with :meth:`save`."""
        path = Path(run_dir) / "metrics.json"
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
