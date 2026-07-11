"""Inference latency and throughput benchmarking.

Measures wall-clock latency of forward passes with proper CUDA
synchronization (async kernel launches otherwise make GPU timings
meaninglessly small) and warmup iterations (the first calls pay one-time
costs: cuDNN autotuning, memory-pool growth, JIT).
"""

from __future__ import annotations

import statistics
import time

import torch
from pydantic import BaseModel, Field
from torch import nn

from e2am.exceptions import ProfilerError
from e2am.utils.logging import get_logger

logger = get_logger("profiler.latency")


class LatencyResult(BaseModel):
    """Latency/throughput statistics for one model + input shape + device."""

    batch_size: int = 1
    iterations: int = 0
    device: str = ""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    throughput_samples_per_s: float = Field(default=0.0, description="batch_size / mean latency.")


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100.0
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = k - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def benchmark_latency(
    model: nn.Module,
    input_size: tuple[int, ...] | None = None,
    sample_input: torch.Tensor | tuple | None = None,
    device: torch.device | str | None = None,
    warmup: int = 10,
    iterations: int = 50,
) -> LatencyResult:
    """Benchmark forward-pass latency.

    Args:
        model: Model to benchmark (left in its current train/eval mode
            afterwards; benchmarked in eval mode without gradients).
        input_size: Full input shape including batch dimension.
        sample_input: Ready-made input tensor/tuple (overrides ``input_size``).
        device: Target device; defaults to the model's parameter device.
        warmup: Untimed warmup iterations.
        iterations: Timed iterations.

    Returns:
        A :class:`LatencyResult` with per-iteration statistics.

    Raises:
        ProfilerError: If no input specification is given or the forward
            pass fails.
    """
    if sample_input is None and input_size is None:
        raise ProfilerError("benchmark_latency needs `input_size` or `sample_input`.")
    if iterations < 1:
        raise ProfilerError("iterations must be >= 1.")

    first_param = next(model.parameters(), None)
    if device is None:
        device = first_param.device if first_param is not None else torch.device("cpu")
    device = torch.device(device)

    if sample_input is None:
        assert input_size is not None
        dtype = first_param.dtype if first_param is not None else torch.float32
        sample_input = torch.randn(*input_size, device=device, dtype=dtype)
    inputs = sample_input if isinstance(sample_input, tuple) else (sample_input,)
    first = inputs[0]
    batch_size = int(first.shape[0]) if isinstance(first, torch.Tensor) and first.dim() else 1

    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    def _sync() -> None:
        if use_cuda:
            torch.cuda.synchronize(device)

    was_training = model.training
    model.eval()
    timings_ms: list[float] = []
    try:
        with torch.no_grad():
            for _ in range(max(warmup, 0)):
                model(*inputs)
            _sync()
            for _ in range(iterations):
                start = time.perf_counter()
                model(*inputs)
                _sync()
                timings_ms.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:
        raise ProfilerError(f"Forward pass failed while benchmarking: {exc}") from exc
    finally:
        model.train(was_training)

    ordered = sorted(timings_ms)
    mean_ms = statistics.fmean(timings_ms)
    result = LatencyResult(
        batch_size=batch_size,
        iterations=iterations,
        device=str(device),
        mean_ms=mean_ms,
        std_ms=statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        min_ms=ordered[0],
        p50_ms=_percentile(ordered, 50.0),
        p95_ms=_percentile(ordered, 95.0),
        max_ms=ordered[-1],
        throughput_samples_per_s=(batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0,
    )
    logger.debug(
        "Latency on %s: mean %.2f ms, p95 %.2f ms, %.1f samples/s",
        result.device,
        result.mean_ms,
        result.p95_ms,
        result.throughput_samples_per_s,
    )
    return result
