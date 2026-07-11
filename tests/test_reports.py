"""Report generation tests: HTML, markdown, PDF, leaderboard, orchestrator."""

import csv
from pathlib import Path

import pytest

from e2am.config.settings import OutputConfig
from e2am.monitoring.result import MonitorResult
from e2am.reports import (
    generate_html_report,
    generate_markdown_report,
    generate_run_artifacts,
    update_leaderboard,
)
from e2am.trainer.result import TrainingResult
from e2am.visualization import generate_monitor_plots


def _training_result(monitor_result: MonitorResult) -> TrainingResult:
    return TrainingResult(
        project="proj",
        run_name="fake-run",
        epochs_requested=3,
        epochs_completed=3,
        samples_processed=300,
        best_val_accuracy=0.8,
        best_epoch=2,
        final_train_loss=0.4,
        final_val_loss=0.5,
        final_val_accuracy=0.8,
        monitor=monitor_result,
        profile={
            "model_name": "TinyNet",
            "params": 1234,
            "params_trainable": 1234,
            "macs": 5678,
            "flops": 11356,
            "model_size_mb": 0.005,
            "coverage": 1.0,
        },
        history={
            "train_loss": {"steps": [0, 1, 2], "values": [1.0, 0.6, 0.4]},
            "val_accuracy": {"steps": [0, 1, 2], "values": [0.5, 0.7, 0.8]},
            "epoch_time_s": {"steps": [0, 1, 2], "values": [1.4, 1.3, 1.3]},
        },
    )


def test_html_report_content(monitor_result: MonitorResult, tmp_path: Path) -> None:
    plots = generate_monitor_plots(monitor_result, tmp_path)
    path = generate_html_report(_training_result(monitor_result), tmp_path, plots)
    text = path.read_text(encoding="utf-8")
    assert path.name == "report.html"
    assert "proj / fake-run" in text
    assert "Experiment summary" in text
    assert "Hardware" in text
    assert "Model profile" in text
    assert "data:image/png;base64," in text  # plots embedded, self-contained
    assert "TinyNet" in text


def test_html_report_for_monitor_only(monitor_result: MonitorResult, tmp_path: Path) -> None:
    path = generate_html_report(monitor_result, tmp_path, [])
    text = path.read_text(encoding="utf-8")
    assert "Energy &amp; utilization" in text
    assert "Model quality" not in text


def test_markdown_report(monitor_result: MonitorResult, tmp_path: Path) -> None:
    plots = generate_monitor_plots(monitor_result, tmp_path)
    path = generate_markdown_report(_training_result(monitor_result), tmp_path, plots)
    text = path.read_text(encoding="utf-8")
    assert path.name == "README.md"
    assert text.startswith("# proj / fake-run")
    assert "## Experiment summary" in text
    assert "![power](power.png)" in text


def test_leaderboard_appends_and_replaces(tmp_path: Path) -> None:
    update_leaderboard({"run_name": "a", "score": 1}, tmp_path)
    update_leaderboard({"run_name": "b", "score": 2}, tmp_path)
    path = update_leaderboard({"run_name": "a", "score": 9}, tmp_path)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {row["run_name"]: row for row in csv.DictReader(handle)}
    assert len(rows) == 2
    assert rows["a"]["score"] == "9"


def test_leaderboard_handles_new_columns(tmp_path: Path) -> None:
    update_leaderboard({"run_name": "a", "score": 1}, tmp_path)
    path = update_leaderboard({"run_name": "b", "extra": "x"}, tmp_path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert set(reader.fieldnames or []) == {"run_name", "score", "extra"}
        rows = list(reader)
    assert rows[0]["extra"] == ""  # old row padded
    assert rows[1]["score"] == ""  # new row padded


def test_pdf_report(monitor_result: MonitorResult, tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    from e2am.reports import generate_pdf_report

    plots = generate_monitor_plots(monitor_result, tmp_path)
    path = generate_pdf_report(_training_result(monitor_result), tmp_path, plots)
    assert path.name == "report.pdf"
    assert path.read_bytes()[:5] == b"%PDF-"
    assert path.stat().st_size > 10_000  # images actually embedded


def test_generate_run_artifacts_full(monitor_result: MonitorResult, tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    result = _training_result(monitor_result)
    output = OutputConfig(save_pdf=True)
    written = generate_run_artifacts(result, tmp_path / "fake-run", output)
    names = {p.name for p in written}
    assert {"report.html", "README.md", "report.pdf", "leaderboard.csv"} <= names
    assert "loss.png" in names
    # leaderboard lands at the results root, next to the run dir
    assert (tmp_path / "leaderboard.csv").exists()


def test_generate_run_artifacts_respects_flags(
    monitor_result: MonitorResult, tmp_path: Path
) -> None:
    output = OutputConfig(save_plots=False, save_html=False, save_csv=False, save_pdf=False)
    written = generate_run_artifacts(monitor_result, tmp_path / "r", output)
    names = {p.name for p in written}
    assert names == {"README.md"}  # markdown is always written
