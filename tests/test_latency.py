"""Latency benchmarking tests."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from e2am.exceptions import ProfilerError  # noqa: E402
from e2am.profiler import benchmark_latency  # noqa: E402

pytestmark = pytest.mark.torch


def test_latency_statistics_are_consistent() -> None:
    model = nn.Linear(32, 32)
    result = benchmark_latency(model, input_size=(8, 32), warmup=2, iterations=20)
    assert result.iterations == 20
    assert result.batch_size == 8
    assert 0 < result.min_ms <= result.p50_ms <= result.p95_ms <= result.max_ms
    assert result.mean_ms > 0
    assert result.throughput_samples_per_s == pytest.approx(8 / (result.mean_ms / 1000.0), rel=1e-6)


def test_training_mode_restored_after_benchmark() -> None:
    model = nn.Linear(4, 4)
    model.train()
    benchmark_latency(model, input_size=(1, 4), warmup=0, iterations=2)
    assert model.training


def test_invalid_iterations_raise() -> None:
    with pytest.raises(ProfilerError, match="iterations"):
        benchmark_latency(nn.Linear(2, 2), input_size=(1, 2), iterations=0)


def test_missing_input_raises() -> None:
    with pytest.raises(ProfilerError, match="input_size"):
        benchmark_latency(nn.Linear(2, 2))


@pytest.mark.gpu
def test_gpu_latency() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = nn.Linear(64, 64).cuda()
    result = benchmark_latency(model, input_size=(16, 64), warmup=3, iterations=10)
    assert result.device.startswith("cuda")
    assert result.mean_ms > 0
