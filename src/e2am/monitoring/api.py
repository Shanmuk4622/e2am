"""The public ``monitor`` API — zero-code-change energy tracking.

Usable as a context manager or a decorator:

    >>> from e2am import monitor
    >>> with monitor(project="ResNet50") as m:
    ...     train()
    >>> print(m.result.total_energy_wh)

    >>> @monitor(project="ResNet50")
    ... def train():
    ...     ...

On exit it stops the session, prints a rich console summary, and (by
default) persists ``metrics.json`` + ``config.yaml`` into
``results/<run_name>/`` so reports can be regenerated later.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table

from e2am.config.settings import ExperimentConfig, MonitorConfig
from e2am.monitoring.result import MonitorResult
from e2am.monitoring.session import MonitorSession
from e2am.utils.logging import get_logger

logger = get_logger("monitor.api")


class monitor(ContextDecorator):
    """Monitor energy, carbon, and utilization around any block of code.

    Args:
        project: Project name grouping related runs.
        run_name: Explicit run identifier; auto-generated when omitted.
        interval_s: Sampling interval in seconds.
        output_dir: Root directory for run artifacts (default ``results``).
        save: Persist ``metrics.json`` and ``config.yaml`` on exit.
        summary: Print a rich console summary on exit.
        config: Full :class:`ExperimentConfig` overriding the simple kwargs.
    """

    def __init__(
        self,
        project: str = "e2am",
        run_name: str | None = None,
        interval_s: float = 1.0,
        output_dir: str | Path | None = None,
        save: bool = True,
        summary: bool = True,
        config: ExperimentConfig | None = None,
    ) -> None:
        if config is None:
            config = ExperimentConfig(
                project=project,
                run_name=run_name,
                monitor=MonitorConfig(sampling_interval_s=interval_s),
            )
            if output_dir is not None:
                config.output.dir = Path(output_dir)
        self.config = config
        self.save = save
        self.summary = summary
        self.run_name = config.resolved_run_name()
        self.result: MonitorResult | None = None
        self._session: MonitorSession | None = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> monitor:
        self._session = MonitorSession(
            config=self.config.monitor,
            project=self.config.project,
            run_name=self.run_name,
        )
        self._session.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        if self._session is None:  # pragma: no cover - defensive
            return False
        self.result = self._session.stop()
        self._session = None
        if exc_type is not None:
            self.result.status = "failed"
        if self.save:
            try:
                self.save_artifacts()
            except Exception as save_exc:
                logger.warning("Could not save run artifacts: %s", save_exc)
        if self.summary:
            self.print_summary()
        return False  # never suppress user exceptions

    # ------------------------------------------------------------------
    # Live access
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Live running totals while the block is executing."""
        if self._session is None:
            raise RuntimeError("monitor() is not active; use it as a context manager.")
        return dict(self._session.snapshot())

    # ------------------------------------------------------------------
    # Persistence & display
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        """Directory holding this run's artifacts."""
        return self.config.output.dir / self.run_name

    def save_artifacts(self) -> Path:
        """Write metrics, config, plots, and reports into the run directory."""
        if self.result is None:
            raise RuntimeError("No result to save; the monitored block has not finished.")
        run_dir = self.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            self.result.model_dump_json(indent=2), encoding="utf-8"
        )
        self.config.to_yaml(run_dir / "config.yaml")
        from e2am.reports.generate import generate_run_artifacts

        generate_run_artifacts(self.result, run_dir, self.config.output)
        logger.info("Run artifacts saved to %s", run_dir)
        return run_dir

    def print_summary(self) -> None:
        """Print a rich summary table of the finished run."""
        if self.result is None:
            return
        r = self.result
        console = Console()
        table = Table(
            title=f"E2AM · {r.project} / {r.run_name}",
            title_style="bold green",
            show_edge=True,
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Status", r.status)
        table.add_row("Duration", f"{r.duration_s:.1f} s")
        table.add_row("Total energy", f"{r.total_energy_wh:.3f} Wh")
        for device in r.devices:
            tag = " (est.)" if device.power_estimated else ""
            table.add_row(
                f"  {device.name} energy{tag}",
                f"{device.energy_wh:.3f} Wh · avg {device.avg_power_w:.1f} W",
            )
        table.add_row(
            "Carbon",
            f"{r.carbon.emissions_g:.3f} g CO2eq "
            f"({r.carbon.intensity_g_per_kwh:.0f} g/kWh, {r.carbon.intensity_source})",
        )
        for name, stats in sorted(r.utilization.items()):
            table.add_row(f"  {name} utilization", f"avg {stats.avg_pct:.1f} %")
        console.print(table)
