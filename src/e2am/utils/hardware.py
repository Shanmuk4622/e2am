"""Hardware detection with energy-capability probing.

Detects CPU, RAM, and GPUs and — crucially for the energy model — probes what
each device can actually report. Machines differ: datacenter GPUs expose real
power sensors via NVML, while consumer cards (e.g. GTX 16xx) only expose
utilization, and most CPUs on Windows expose no power interface at all. The
resulting :class:`SystemInfo` records these capabilities so the monitor
module can choose between *measured* power and *estimated* power
(TDP × utilization) per device.
"""

from __future__ import annotations

import platform
import socket
import sys
from typing import Any

import psutil
from pydantic import BaseModel, Field

from e2am.utils.logging import get_logger
from e2am.utils.nvml import NVML

logger = get_logger("utils.hardware")

#: Fallback CPU TDP when nothing better is known (typical desktop CPU).
DEFAULT_CPU_TDP_W = 65.0

#: Fallback GPU board power when NVML reports no power limit.
DEFAULT_GPU_TDP_W = 150.0

#: RAM power estimate: ~3 W per 8 GB DIMM (CodeCarbon's convention).
RAM_WATTS_PER_GB = 3.0 / 8.0


class GPUInfo(BaseModel):
    """Static description of one GPU and its measurement capabilities."""

    index: int
    name: str = "Unknown GPU"
    total_memory_mb: float | None = None
    driver_version: str | None = None
    power_limit_w: float | None = Field(
        default=None,
        description="Enforced board power limit in watts (TDP proxy).",
    )
    supports_power_reading: bool = Field(
        default=False,
        description="Whether NVML exposes a live power sensor for this card.",
    )
    supports_utilization: bool = Field(
        default=False,
        description="Whether NVML exposes utilization rates for this card.",
    )

    @property
    def tdp_w(self) -> float:
        """Best-known board power budget, used for power estimation."""
        return self.power_limit_w if self.power_limit_w else DEFAULT_GPU_TDP_W


class CPUInfo(BaseModel):
    """Static description of the CPU."""

    model: str = "Unknown CPU"
    physical_cores: int | None = None
    logical_cores: int | None = None
    max_frequency_mhz: float | None = None
    tdp_w: float = Field(
        default=DEFAULT_CPU_TDP_W,
        description="Thermal design power used for energy estimation.",
    )


class RAMInfo(BaseModel):
    """Static description of system memory."""

    total_gb: float = 0.0
    estimated_power_w: float = Field(
        default=0.0,
        description="Estimated full-load RAM power (~3 W per 8 GB).",
    )


class SystemInfo(BaseModel):
    """Full snapshot of the machine E2AM is running on."""

    os: str = ""
    os_version: str = ""
    python_version: str = ""
    hostname: str = ""
    cpu: CPUInfo = Field(default_factory=CPUInfo)
    ram: RAMInfo = Field(default_factory=RAMInfo)
    gpus: list[GPUInfo] = Field(default_factory=list)
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None

    @property
    def gpu_count(self) -> int:
        """Number of detected NVIDIA GPUs."""
        return len(self.gpus)


def _detect_cpu(cpu_tdp_w: float | None) -> CPUInfo:
    model = platform.processor() or platform.machine() or "Unknown CPU"
    freq = None
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq is not None and cpu_freq.max:
            freq = float(cpu_freq.max)
    except Exception as exc:  # some platforms raise here
        logger.debug("psutil.cpu_freq failed: %s", exc)
    return CPUInfo(
        model=model,
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        max_frequency_mhz=freq,
        tdp_w=cpu_tdp_w if cpu_tdp_w is not None else DEFAULT_CPU_TDP_W,
    )


def _detect_ram() -> RAMInfo:
    total_gb = psutil.virtual_memory().total / 2**30
    return RAMInfo(
        total_gb=round(total_gb, 2), estimated_power_w=round(total_gb * RAM_WATTS_PER_GB, 2)
    )


def _detect_gpus() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    with NVML() as nvml:
        if not nvml.available:
            return gpus
        driver = nvml.driver_version()
        for index, handle in enumerate(nvml.handles()):
            mem = nvml.memory_info(handle)
            gpus.append(
                GPUInfo(
                    index=index,
                    name=nvml.device_name(handle) or "Unknown GPU",
                    total_memory_mb=mem[1] if mem else None,
                    driver_version=driver,
                    power_limit_w=nvml.power_limit_w(handle),
                    supports_power_reading=nvml.power_usage_w(handle) is not None,
                    supports_utilization=nvml.utilization(handle) is not None,
                )
            )
    return gpus


def _import_torch() -> Any:
    """Return the torch module without forcing a slow import when absent.

    Only imports torch if it is already loaded or installed; never raises.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        return torch
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return None
        import torch as torch_module

        return torch_module
    except Exception as exc:
        logger.debug("torch inspection failed: %s", exc)
        return None


def _detect_torch() -> tuple[str | None, bool, str | None]:
    """Return (torch_version, cuda_available, cuda_version); never raises."""
    torch = _import_torch()
    if torch is None:
        return None, False, None
    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        return str(torch.__version__), cuda_available, cuda_version
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("torch CUDA inspection failed: %s", exc)
        return str(getattr(torch, "__version__", None)), False, None


def detect_hardware(cpu_tdp_w: float | None = None) -> SystemInfo:
    """Detect the host system's hardware and measurement capabilities.

    Args:
        cpu_tdp_w: Override for the CPU's thermal design power in watts. When
            omitted, a conservative desktop default (65 W) is used, since most
            operating systems expose no portable CPU power interface.

    Returns:
        A fully populated :class:`SystemInfo`.
    """
    torch_version, cuda_available, cuda_version = _detect_torch()
    info = SystemInfo(
        os=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        hostname=socket.gethostname(),
        cpu=_detect_cpu(cpu_tdp_w),
        ram=_detect_ram(),
        gpus=_detect_gpus(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
    )
    for gpu in info.gpus:
        if not gpu.supports_power_reading:
            logger.debug(
                "GPU %d (%s) has no power sensor; energy will be estimated "
                "as TDP (%.0f W) x utilization.",
                gpu.index,
                gpu.name,
                gpu.tdp_w,
            )
    return info
