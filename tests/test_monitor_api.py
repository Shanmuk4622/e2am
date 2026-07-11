"""Tests for the public monitor() context manager / decorator."""

import json
import time
from pathlib import Path

import pytest

from e2am import monitor


def test_context_manager_produces_result(tmp_path: Path) -> None:
    with monitor(
        project="demo",
        run_name="ctx-run",
        interval_s=0.05,
        output_dir=tmp_path,
        summary=False,
    ) as m:
        time.sleep(0.2)
    assert m.result is not None
    assert m.result.status == "completed"
    assert m.result.total_energy_j > 0
    assert m.result.duration_s > 0


def test_artifacts_written(tmp_path: Path) -> None:
    with monitor(
        project="demo",
        run_name="artifacts-run",
        interval_s=0.05,
        output_dir=tmp_path,
        summary=False,
    ):
        time.sleep(0.15)
    run_dir = tmp_path / "artifacts-run"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_name"] == "artifacts-run"
    assert (run_dir / "config.yaml").exists()


def test_decorator_usage(tmp_path: Path) -> None:
    wrapper = monitor(
        project="demo",
        run_name="deco-run",
        interval_s=0.05,
        output_dir=tmp_path,
        summary=False,
    )

    @wrapper
    def work() -> int:
        time.sleep(0.15)
        return 42

    assert work() == 42
    assert wrapper.result is not None
    assert wrapper.result.run_name == "deco-run"


def test_exception_propagates_and_marks_failed(tmp_path: Path) -> None:
    m = monitor(
        project="demo",
        run_name="fail-run",
        interval_s=0.05,
        output_dir=tmp_path,
        summary=False,
    )
    with pytest.raises(ValueError, match="boom"), m:
        time.sleep(0.1)
        raise ValueError("boom")
    assert m.result is not None
    assert m.result.status == "failed"
    # artifacts are still written for failed runs
    assert (tmp_path / "fail-run" / "metrics.json").exists()


def test_snapshot_only_valid_inside_block(tmp_path: Path) -> None:
    m = monitor(output_dir=tmp_path, summary=False, save=False, interval_s=0.05)
    with pytest.raises(RuntimeError):
        m.snapshot()
    with m:
        snap = m.snapshot()
        assert "total_energy_j" in snap


def test_auto_run_name(tmp_path: Path) -> None:
    m = monitor(project="autoname", output_dir=tmp_path, summary=False, interval_s=0.05)
    assert m.run_name.startswith("autoname-")
