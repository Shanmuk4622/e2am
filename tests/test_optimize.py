"""Optimization analyzer tests — each rule verified with exact expectations."""

from datetime import datetime, timezone

import pytest

from e2am.monitoring.result import DeviceEnergy, MonitorResult, UtilizationStats
from e2am.optimize import analyze
from e2am.optimize.analyzer import (
    rule_gpu_underutilized,
    rule_memory_pressure,
    rule_mixed_precision,
    rule_quantized_inference,
    rule_torch_compile,
    rule_wasted_epochs,
)
from e2am.trainer.result import TrainingResult
from e2am.utils.hardware import GPUInfo, SystemInfo


def _monitor(
    gpu_util: float = 90.0,
    cpu_util: float = 30.0,
    torch_version: str = "2.5.1",
    gpu_total_mb: float = 4096.0,
    gpu_peak_mb: float = 1000.0,
    gpu_energy_j: float = 3600.0,
) -> MonitorResult:
    now = datetime.now(timezone.utc)
    return MonitorResult(
        started_at=now,
        ended_at=now,
        duration_s=100.0,
        system=SystemInfo(
            torch_version=torch_version,
            cuda_available=True,
            gpus=[GPUInfo(index=0, name="Test GPU", total_memory_mb=gpu_total_mb)],
        ),
        devices=[DeviceEnergy(name="gpu0", energy_j=gpu_energy_j)],
        utilization={
            "gpu0": UtilizationStats(avg_pct=gpu_util, max_pct=100.0),
            "cpu": UtilizationStats(avg_pct=cpu_util, max_pct=100.0),
        },
        peak_memory_mb={"gpu0": gpu_peak_mb},
    )


def _result(**kwargs) -> TrainingResult:
    defaults = {
        "project": "p",
        "run_name": "r",
        "device": "cuda",
        "mixed_precision": True,
        "epochs_completed": 5,
        "monitor": _monitor(),
    }
    defaults.update(kwargs)
    return TrainingResult(**defaults)


def test_mixed_precision_fires_only_without_amp_on_cuda() -> None:
    assert rule_mixed_precision(_result(mixed_precision=True)) is None
    assert rule_mixed_precision(_result(device="cpu", mixed_precision=False)) is None
    suggestion = rule_mixed_precision(_result(mixed_precision=False))
    assert suggestion is not None
    assert suggestion.priority == "high"
    # 25% of 1 Wh of GPU energy
    assert suggestion.estimated_savings_wh == pytest.approx(0.25)


def test_wasted_epochs_quantified_exactly() -> None:
    result = _result(
        history={
            "val_accuracy": {"steps": [0, 1, 2, 3, 4], "values": [0.5, 0.8, 0.9, 0.9, 0.9]},
            "cumulative_energy_wh": {
                "steps": [0, 1, 2, 3, 4],
                "values": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
        }
    )
    suggestion = rule_wasted_epochs(result)
    assert suggestion is not None
    # converged at epoch index 2 (energy 3.0); wasted = 5.0 - 3.0
    assert suggestion.estimated_savings_wh == pytest.approx(2.0)
    assert "2 epoch(s)" in suggestion.title


def test_wasted_epochs_silent_when_still_improving() -> None:
    result = _result(
        history={
            "val_accuracy": {"steps": [0, 1, 2], "values": [0.5, 0.7, 0.9]},
            "cumulative_energy_wh": {"steps": [0, 1, 2], "values": [1.0, 2.0, 3.0]},
        }
    )
    assert rule_wasted_epochs(result) is None


def test_wasted_epochs_needs_history() -> None:
    assert rule_wasted_epochs(_result(history={})) is None


def test_gpu_underutilized_distinguishes_input_bottleneck() -> None:
    starved = _result(monitor=_monitor(gpu_util=20.0, cpu_util=95.0))
    suggestion = rule_gpu_underutilized(starved)
    assert suggestion is not None
    assert "num_workers" in suggestion.action

    small_batches = _result(monitor=_monitor(gpu_util=20.0, cpu_util=30.0))
    suggestion = rule_gpu_underutilized(small_batches)
    assert suggestion is not None
    assert "batch_size" in suggestion.action

    busy = _result(monitor=_monitor(gpu_util=85.0))
    assert rule_gpu_underutilized(busy) is None
    assert rule_gpu_underutilized(_result(device="cpu")) is None


def test_torch_compile_only_on_torch2_cuda() -> None:
    assert rule_torch_compile(_result(monitor=_monitor(torch_version="1.13.1"))) is None
    assert rule_torch_compile(_result(device="cpu")) is None
    suggestion = rule_torch_compile(_result())
    assert suggestion is not None
    assert "torch.compile" in suggestion.action


def test_memory_pressure_threshold() -> None:
    relaxed = _result(monitor=_monitor(gpu_peak_mb=1000.0, gpu_total_mb=4096.0))
    assert rule_memory_pressure(relaxed) is None
    squeezed = _result(monitor=_monitor(gpu_peak_mb=3900.0, gpu_total_mb=4096.0))
    suggestion = rule_memory_pressure(squeezed)
    assert suggestion is not None
    assert "checkpointing" in suggestion.title.lower()


def test_quantization_only_for_large_models() -> None:
    small = _result(profile={"params": 10_000})
    assert rule_quantized_inference(small) is None
    large = _result(profile={"params": 5_000_000})
    suggestion = rule_quantized_inference(large)
    assert suggestion is not None
    assert "5,000,000" in suggestion.rationale


def test_analyze_sorts_by_priority_and_survives_bad_data() -> None:
    result = _result(
        mixed_precision=False,  # high
        profile={"params": 5_000_000},  # low
        monitor=_monitor(gpu_util=90.0),
    )
    suggestions = analyze(result)
    ids = [s.id for s in suggestions]
    assert "mixed-precision" in ids
    assert "quantized-inference" in ids
    priorities = [s.priority for s in suggestions]
    assert priorities == sorted(priorities, key=lambda p: {"high": 0, "medium": 1, "low": 2}[p])


def test_efficient_run_yields_no_high_priority_noise() -> None:
    result = _result()  # AMP on, GPU busy, small model, no history
    ids = {s.id for s in analyze(result)}
    assert "mixed-precision" not in ids
    assert "gpu-underutilized" not in ids
    assert "early-stopping" not in ids
