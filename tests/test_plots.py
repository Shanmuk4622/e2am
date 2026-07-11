"""Plot generation tests — verify files, formats, and graceful skipping."""

from pathlib import Path

import pytest

from e2am.monitoring.result import MonitorResult
from e2am.trainer.result import TrainingResult
from e2am.visualization import generate_monitor_plots, generate_plots

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(path: Path) -> None:
    assert path.exists(), f"missing {path}"
    assert path.read_bytes()[:8] == PNG_MAGIC, f"not a PNG: {path}"


def test_monitor_plots_written(monitor_result: MonitorResult, tmp_path: Path) -> None:
    written = generate_monitor_plots(monitor_result, tmp_path)
    names = {p.name for p in written}
    assert {
        "power.png",
        "energy.png",
        "gpu_usage.png",
        "cpu_usage.png",
        "memory.png",
        "carbon.png",
    } <= names
    for path in written:
        _assert_valid_png(path)


def test_empty_monitor_result_writes_nothing(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    empty = MonitorResult(
        started_at=datetime.now(timezone.utc), ended_at=datetime.now(timezone.utc)
    )
    assert generate_monitor_plots(empty, tmp_path) == []


def test_training_plots_written(monitor_result: MonitorResult, tmp_path: Path) -> None:
    result = TrainingResult(
        project="proj",
        run_name="fake-run",
        epochs_requested=3,
        epochs_completed=3,
        samples_processed=300,
        monitor=monitor_result,
        history={
            "train_loss": {"steps": [0, 1, 2], "values": [1.0, 0.6, 0.4]},
            "val_loss": {"steps": [0, 1, 2], "values": [1.1, 0.7, 0.5]},
            "val_accuracy": {"steps": [0, 1, 2], "values": [0.5, 0.7, 0.8]},
            "epoch_time_s": {"steps": [0, 1, 2], "values": [1.4, 1.3, 1.3]},
        },
    )
    written = generate_plots(result, tmp_path)
    names = {p.name for p in written}
    assert {"loss.png", "accuracy.png", "latency.png", "throughput.png", "power.png"} <= names
    for path in written:
        _assert_valid_png(path)


def test_training_plots_without_monitor_or_history(tmp_path: Path) -> None:
    result = TrainingResult(project="p", run_name="r")
    assert generate_plots(result, tmp_path) == []


def test_cpu_only_run_skips_gpu_plot(monitor_result: MonitorResult, tmp_path: Path) -> None:
    ts = monitor_result.timeseries
    for channel_map in (ts.power_w, ts.utilization_pct, ts.memory_used_mb, ts.temperature_c):
        channel_map.pop("gpu0", None)
    monitor_result.devices = [d for d in monitor_result.devices if d.name != "gpu0"]
    written = generate_monitor_plots(monitor_result, tmp_path)
    names = {p.name for p in written}
    assert "gpu_usage.png" not in names
    assert "power.png" in names


def test_plot_regeneration_overwrites(monitor_result: MonitorResult, tmp_path: Path) -> None:
    first = generate_monitor_plots(monitor_result, tmp_path)
    second = generate_monitor_plots(monitor_result, tmp_path)
    assert {p.name for p in first} == {p.name for p in second}


@pytest.mark.parametrize("channel", ["cpu", "gpu0"])
def test_all_none_channel_is_skipped(
    monitor_result: MonitorResult, tmp_path: Path, channel: str
) -> None:
    monitor_result.timeseries.power_w[channel] = [None] * 5
    written = generate_monitor_plots(monitor_result, tmp_path)
    assert any(p.name == "power.png" for p in written)  # other channels still plot
