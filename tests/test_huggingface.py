"""Hugging Face integration tests — drive the callback like HF Trainer does."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from e2am.integrations import E2AMCallback  # noqa: E402

pytestmark = pytest.mark.torch


def _hf_args() -> SimpleNamespace:
    return SimpleNamespace(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        fp16=True,
        bf16=False,
    )


def _drive_training(callback: E2AMCallback) -> None:
    """Simulate the call sequence HF Trainer makes during a 2-epoch run."""
    args, control = _hf_args(), SimpleNamespace()
    state = SimpleNamespace(global_step=0, epoch=0.0)

    callback.on_train_begin(args, state, control)
    time.sleep(0.15)

    state.global_step, state.epoch = 10, 1.0
    callback.on_log(args, state, control, logs={"loss": 0.9, "learning_rate": 1e-4})
    callback.on_log(args, state, control, logs={"eval_loss": 0.8, "eval_accuracy": 0.7})
    callback.on_epoch_end(args, state, control)
    time.sleep(0.1)

    state.global_step, state.epoch = 20, 2.0
    callback.on_log(args, state, control, logs={"loss": 0.5})
    callback.on_log(args, state, control, logs={"eval_loss": 0.4, "eval_accuracy": 0.9})
    callback.on_epoch_end(args, state, control)
    callback.on_train_end(args, state, control)


def test_callback_is_a_real_trainer_callback() -> None:
    from transformers import TrainerCallback

    assert issubclass(E2AMCallback, TrainerCallback)


def test_full_lifecycle_builds_training_result(tmp_path: Path) -> None:
    callback = E2AMCallback(
        project="hf-demo",
        run_name="hf-run",
        output_dir=tmp_path,
        sampling_interval_s=0.05,
    )
    _drive_training(callback)

    result = callback.result
    assert result is not None
    assert result.project == "hf-demo"
    assert result.epochs_completed == 2
    assert result.samples_processed == 20 * 8 * 2  # steps x batch x accumulation
    assert result.mixed_precision is True
    assert result.best_val_accuracy == pytest.approx(0.9)
    assert result.final_train_loss == pytest.approx(0.5)
    assert result.final_val_loss == pytest.approx(0.4)
    assert result.monitor is not None and result.monitor.total_energy_j > 0
    assert result.green is not None
    assert result.green.green_score is not None
    assert result.green.eag_pct_per_wh is not None  # two eval points recorded
    # HF log streams landed in history
    assert "loss" in result.history
    assert "cumulative_energy_wh" in result.history


def test_artifacts_written_for_hf_run(tmp_path: Path) -> None:
    callback = E2AMCallback(
        project="hf-demo",
        run_name="hf-artifacts",
        output_dir=tmp_path,
        sampling_interval_s=0.05,
    )
    _drive_training(callback)
    run_dir = tmp_path / "hf-artifacts"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_name"] == "hf-artifacts"
    assert (run_dir / "report.html").exists()
    assert (run_dir / "config.yaml").exists()
    assert (tmp_path / "leaderboard.csv").exists()


def test_non_numeric_and_empty_logs_ignored(tmp_path: Path) -> None:
    callback = E2AMCallback(output_dir=tmp_path, sampling_interval_s=0.05, save_artifacts=False)
    args, control = _hf_args(), SimpleNamespace()
    state = SimpleNamespace(global_step=1, epoch=0.5)
    callback.on_train_begin(args, state, control)
    callback.on_log(args, state, control, logs=None)
    callback.on_log(args, state, control, logs={"note": "warmup done", "loss": 1.0})
    callback.on_train_end(args, state, control)
    assert callback.result is not None
    assert "note" not in callback.result.history
    assert callback.result.final_train_loss == pytest.approx(1.0)


def test_train_end_without_begin_is_safe(tmp_path: Path) -> None:
    callback = E2AMCallback(output_dir=tmp_path, save_artifacts=False)
    callback.on_train_end(_hf_args(), SimpleNamespace(global_step=0, epoch=0), SimpleNamespace())
    assert callback.result is None
