"""CLI tests via typer's CliRunner (no subprocesses, real code paths)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from e2am.cli.main import app
from e2am.monitoring.result import MonitorResult

runner = CliRunner()

SCRIPT = """
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_model():
    return nn.Linear(8, 2)


def get_loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 8, generator=g)
    y = (x.sum(dim=1) > 0).long()
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16), DataLoader(ds, batch_size=32)
"""


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "e2am" in result.output


def test_hardware_table_and_json() -> None:
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "CPU" in result.output
    result = runner.invoke(app, ["hardware", "--json"])
    assert result.exit_code == 0
    assert '"cpu"' in result.output


@pytest.mark.torch
def test_train_command(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from e2am.config.settings import ExperimentConfig

    script = tmp_path / "model.py"
    script.write_text(SCRIPT, encoding="utf-8")
    config = ExperimentConfig(project="cli-test", run_name="cli-run")
    config.trainer.epochs = 1
    config.trainer.device = "cpu"
    config.monitor.sampling_interval_s = 0.05
    config.output.dir = tmp_path / "results"
    config_path = config.to_yaml(tmp_path / "config.yaml")

    result = runner.invoke(app, ["train", str(script), "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "Run complete" in result.output
    run_dir = tmp_path / "results" / "cli-run"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "report.html").exists()


@pytest.mark.torch
def test_train_missing_factory_fails_cleanly(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    script = tmp_path / "broken.py"
    script.write_text("x = 1\n", encoding="utf-8")
    result = runner.invoke(app, ["train", str(script)])
    assert result.exit_code == 1
    assert "get_model" in result.output


@pytest.mark.torch
def test_benchmark_command(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    script = tmp_path / "model.py"
    script.write_text(SCRIPT, encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(script),
            "--input-size",
            "4,8",
            "--iterations",
            "5",
            "--warmup",
            "1",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Parameters" in result.output
    assert "Energy / inference" in result.output


@pytest.mark.torch
def test_benchmark_bad_input_size(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    script = tmp_path / "model.py"
    script.write_text(SCRIPT, encoding="utf-8")
    result = runner.invoke(app, ["benchmark", str(script), "--input-size", "abc"])
    assert result.exit_code == 1
    assert "comma-separated" in result.output


def test_report_command(monitor_result: MonitorResult, tmp_path: Path) -> None:
    run_dir = tmp_path / "fake-run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(monitor_result.model_dump_json(), encoding="utf-8")
    result = runner.invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert (run_dir / "report.html").exists()
    assert (run_dir / "README.md").exists()
    assert (tmp_path / "leaderboard.csv").exists()


def test_compare_command(monitor_result: MonitorResult, tmp_path: Path) -> None:
    for name in ("run-a", "run-b"):
        run_dir = tmp_path / name
        run_dir.mkdir()
        clone = monitor_result.model_copy(deep=True)
        clone.run_name = name
        (run_dir / "metrics.json").write_text(clone.model_dump_json(), encoding="utf-8")
    result = runner.invoke(app, ["compare", str(tmp_path / "run-a"), str(tmp_path / "run-b")])
    assert result.exit_code == 0, result.output
    assert "run-a" in result.output
    assert "run-b" in result.output
    assert "total_energy_wh" in result.output


def test_compare_needs_two_runs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compare", str(tmp_path)])
    assert result.exit_code == 1


def test_dashboard_command(monitor_result: MonitorResult, tmp_path: Path) -> None:
    from e2am.reports.leaderboard import update_leaderboard

    update_leaderboard(monitor_result.to_flat_dict(), tmp_path)
    result = runner.invoke(app, ["dashboard", str(tmp_path), "--no-open"])
    assert result.exit_code == 0, result.output
    dashboard = tmp_path / "dashboard.html"
    assert dashboard.exists()
    assert "fake-run" in dashboard.read_text(encoding="utf-8")


def test_dashboard_without_runs_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["dashboard", str(tmp_path), "--no-open"])
    assert result.exit_code == 1
    assert "leaderboard" in result.output.lower()
