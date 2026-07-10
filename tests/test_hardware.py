"""Hardware detection tests — must pass on any machine, GPU or not."""

import json

from e2am.utils.hardware import (
    DEFAULT_CPU_TDP_W,
    CPUInfo,
    GPUInfo,
    SystemInfo,
    detect_hardware,
)


def test_detect_hardware_returns_system_info() -> None:
    info = detect_hardware()
    assert isinstance(info, SystemInfo)
    assert info.os
    assert info.python_version
    assert info.cpu.logical_cores is None or info.cpu.logical_cores >= 1
    assert info.ram.total_gb > 0


def test_cpu_tdp_default_and_override() -> None:
    assert detect_hardware().cpu.tdp_w == DEFAULT_CPU_TDP_W
    assert detect_hardware(cpu_tdp_w=95.0).cpu.tdp_w == 95.0


def test_system_info_is_json_serializable() -> None:
    info = detect_hardware()
    payload = json.loads(info.model_dump_json())
    assert payload["os"] == info.os
    assert isinstance(payload["gpus"], list)


def test_gpu_info_tdp_fallback() -> None:
    gpu = GPUInfo(index=0, name="Test GPU")
    assert gpu.tdp_w > 0  # falls back to default when no power limit known
    gpu_with_limit = GPUInfo(index=0, name="Test GPU", power_limit_w=75.0)
    assert gpu_with_limit.tdp_w == 75.0


def test_cpu_info_defaults() -> None:
    cpu = CPUInfo()
    assert cpu.tdp_w == DEFAULT_CPU_TDP_W


def test_ram_power_estimate_positive() -> None:
    info = detect_hardware()
    assert info.ram.estimated_power_w > 0
