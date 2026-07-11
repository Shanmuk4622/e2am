"""Device samplers — the Strategy layer of the monitor.

Each sampler wraps one power/utilization source (CPU, RAM, one GPU) behind a
uniform interface. Samplers must *never* raise from :meth:`Sampler.sample`;
a failed reading returns an empty :class:`Sample` so a flaky sensor cannot
crash a multi-hour training run.

Power readings are tagged as measured or estimated:

* GPUs with an NVML power sensor report **measured** board power.
* GPUs without one (e.g. many GTX 16xx cards) report **estimated** power as
  ``board_power_limit × utilization``.
* CPUs and RAM are always **estimated** (``TDP × utilization`` and
  ``~3 W per 8 GB used``), since no portable OS interface exposes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import psutil

from e2am.config.settings import MonitorConfig
from e2am.utils.hardware import DEFAULT_CPU_TDP_W, DEFAULT_GPU_TDP_W, RAM_WATTS_PER_GB
from e2am.utils.logging import get_logger
from e2am.utils.nvml import NVML, NVMLHandle

logger = get_logger("monitor.samplers")


@dataclass
class Sample:
    """One point-in-time reading from a device. Missing values are ``None``."""

    power_w: float | None = None
    utilization_pct: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None


class Sampler(ABC):
    """Interface for one monitored device."""

    def __init__(self, name: str) -> None:
        #: Channel name used in results and plots (e.g. ``"cpu"``, ``"gpu0"``).
        self.name = name

    @property
    @abstractmethod
    def estimates_power(self) -> bool:
        """``True`` when power is estimated rather than sensor-measured."""

    @abstractmethod
    def setup(self) -> bool:
        """Prepare the sampler. Returns ``False`` if the device is unusable."""

    @abstractmethod
    def sample(self) -> Sample:
        """Take one reading. Must not raise; return empty fields on failure."""

    def teardown(self) -> None:  # noqa: B027 - optional hook, deliberately non-abstract
        """Release resources. Never raises."""


class CPUSampler(Sampler):
    """Whole-package CPU sampler using ``TDP × utilization`` estimation."""

    def __init__(self, tdp_w: float = DEFAULT_CPU_TDP_W) -> None:
        super().__init__("cpu")
        self.tdp_w = tdp_w

    @property
    def estimates_power(self) -> bool:
        return True

    def setup(self) -> bool:
        try:
            # First cpu_percent() call always returns 0.0; prime it so real
            # readings start with the very first tick.
            psutil.cpu_percent(interval=None)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("CPU sampler setup failed: %s", exc)
            return False

    def sample(self) -> Sample:
        try:
            util = float(psutil.cpu_percent(interval=None))
            return Sample(power_w=self.tdp_w * util / 100.0, utilization_pct=util)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("CPU sample failed: %s", exc)
            return Sample()


class RAMSampler(Sampler):
    """System-memory sampler using the ~3 W per 8 GB *used* convention."""

    def __init__(self, watts_per_gb: float = RAM_WATTS_PER_GB) -> None:
        super().__init__("ram")
        self.watts_per_gb = watts_per_gb

    @property
    def estimates_power(self) -> bool:
        return True

    def setup(self) -> bool:
        try:
            psutil.virtual_memory()
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("RAM sampler setup failed: %s", exc)
            return False

    def sample(self) -> Sample:
        try:
            vm = psutil.virtual_memory()
            used_gb = (vm.total - vm.available) / 2**30
            return Sample(
                power_w=used_gb * self.watts_per_gb,
                utilization_pct=float(vm.percent),
                memory_used_mb=used_gb * 1024.0,
                memory_total_mb=vm.total / 2**20,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("RAM sample failed: %s", exc)
            return Sample()


class GPUSampler(Sampler):
    """NVML-backed sampler for one GPU, with estimation fallback.

    Uses the live power sensor when the card exposes one; otherwise estimates
    ``board_power_limit × utilization``. Each instance owns its own NVML
    session (NVML init/shutdown is reference-counted by the driver).
    """

    def __init__(self, index: int, fallback_tdp_w: float = DEFAULT_GPU_TDP_W) -> None:
        super().__init__(f"gpu{index}")
        self.index = index
        self.fallback_tdp_w = fallback_tdp_w
        self._nvml = NVML()
        self._handle: NVMLHandle | None = None
        self._has_power_sensor = False
        self._tdp_w = fallback_tdp_w

    @property
    def estimates_power(self) -> bool:
        return not self._has_power_sensor

    @property
    def tdp_w(self) -> float:
        """Board power budget used when estimating."""
        return self._tdp_w

    def setup(self) -> bool:
        if not self._nvml.init():
            return False
        handles = self._nvml.handles([self.index])
        if not handles:
            logger.debug("GPU %d not found via NVML.", self.index)
            self._nvml.shutdown()
            return False
        self._handle = handles[0]
        self._has_power_sensor = self._nvml.power_usage_w(self._handle) is not None
        limit = self._nvml.power_limit_w(self._handle)
        if limit:
            self._tdp_w = limit
        if not self._has_power_sensor:
            logger.info(
                "GPU %d (%s) has no power sensor; estimating power as " "%.0f W x utilization.",
                self.index,
                self._nvml.device_name(self._handle) or "unknown",
                self._tdp_w,
            )
        return True

    def sample(self) -> Sample:
        if self._handle is None:
            return Sample()
        power = self._nvml.power_usage_w(self._handle)
        util = self._nvml.utilization(self._handle)
        mem = self._nvml.memory_info(self._handle)
        util_pct = util[0] if util is not None else None
        if power is None and util_pct is not None:
            power = self._tdp_w * util_pct / 100.0
        return Sample(
            power_w=power,
            utilization_pct=util_pct,
            memory_used_mb=mem[0] if mem else None,
            memory_total_mb=mem[1] if mem else None,
            temperature_c=self._nvml.temperature_c(self._handle),
        )

    def teardown(self) -> None:
        self._handle = None
        self._nvml.shutdown()


def create_samplers(config: MonitorConfig) -> list[Sampler]:
    """Build and set up all samplers available on this machine.

    Args:
        config: Monitoring configuration (GPU selection, CPU TDP override).

    Returns:
        Ready-to-use samplers; devices that fail setup are silently skipped
        (with a debug log), so this works on GPU-less machines too.
    """
    cpu_tdp = config.cpu_tdp_w if config.cpu_tdp_w is not None else DEFAULT_CPU_TDP_W
    candidates: list[Sampler] = [CPUSampler(tdp_w=cpu_tdp), RAMSampler()]

    with NVML() as nvml:
        gpu_count = nvml.device_count()
    indices = config.gpu_indices if config.gpu_indices is not None else list(range(gpu_count))
    candidates.extend(GPUSampler(i) for i in indices)

    ready: list[Sampler] = []
    for sampler in candidates:
        try:
            if sampler.setup():
                ready.append(sampler)
            else:
                logger.debug("Sampler %s unavailable; skipping.", sampler.name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Sampler %s setup raised: %s", sampler.name, exc)
    return ready
