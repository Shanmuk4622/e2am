"""Model profiling: FLOPs/MACs/params, latency/throughput, peak memory.

This package requires PyTorch (it profiles torch models); it is imported
lazily by the rest of E2AM so monitoring-only installs never touch it.
"""

from __future__ import annotations

from e2am.profiler.flops import ModelProfile, profile_model, register_mac_counter
from e2am.profiler.latency import LatencyResult, benchmark_latency
from e2am.profiler.memory import MemoryTracker, MemoryUsage

__all__ = [
    "LatencyResult",
    "MemoryTracker",
    "MemoryUsage",
    "ModelProfile",
    "benchmark_latency",
    "profile_model",
    "register_mac_counter",
]
