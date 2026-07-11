"""Thin, failure-tolerant wrapper around NVIDIA NVML (via ``nvidia-ml-py``).

NVML availability varies wildly across machines: no NVIDIA GPU, missing
driver, or consumer cards (e.g. GTX 16xx series) that expose utilization but
*not* power sensors. Every accessor here returns ``None`` instead of raising,
so callers can build graceful fallbacks (e.g. TDP × utilization power
estimation) without try/except noise.

Usage:
    >>> nvml = NVML()
    >>> if nvml.init():
    ...     for handle in nvml.handles():
    ...         print(nvml.device_name(handle), nvml.power_usage_w(handle))
    ...     nvml.shutdown()
"""

from __future__ import annotations

from typing import Any

from e2am.utils.logging import get_logger

try:  # pragma: no cover - import success depends on the machine
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:  # pragma: no cover
    pynvml = None
    _PYNVML_AVAILABLE = False

logger = get_logger("utils.nvml")

# NVML handles are opaque C pointers; expose them as Any.
NVMLHandle = Any


class NVML:
    """Safe NVML session wrapper with graceful degradation.

    All query methods return ``None`` (or empty collections) when NVML is
    unavailable or the specific sensor is unsupported on the device.
    """

    def __init__(self) -> None:
        self._initialized = False

    @property
    def available(self) -> bool:
        """Whether NVML has been successfully initialized."""
        return self._initialized

    def init(self) -> bool:
        """Initialize NVML. Safe to call multiple times.

        Returns:
            ``True`` if NVML is ready to use, ``False`` otherwise.
        """
        if self._initialized:
            return True
        if not _PYNVML_AVAILABLE:
            logger.debug("pynvml is not installed; GPU monitoring disabled.")
            return False
        try:
            pynvml.nvmlInit()
            self._initialized = True
            return True
        except Exception as exc:  # NVMLError, driver missing, etc.
            logger.debug("NVML initialization failed: %s", exc)
            return False

    def shutdown(self) -> None:
        """Shut down NVML if it was initialized. Never raises."""
        if not self._initialized:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("NVML shutdown failed: %s", exc)
        finally:
            self._initialized = False

    def __enter__(self) -> NVML:
        self.init()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Queries (all None-on-failure)
    # ------------------------------------------------------------------

    def device_count(self) -> int:
        """Number of NVML-visible GPUs (0 when NVML is unavailable)."""
        if not self._initialized:
            return 0
        try:
            return int(pynvml.nvmlDeviceGetCount())
        except Exception:
            return 0

    def handles(self, indices: list[int] | None = None) -> list[NVMLHandle]:
        """Device handles for the given indices (default: all devices)."""
        if not self._initialized:
            return []
        count = self.device_count()
        wanted = range(count) if indices is None else [i for i in indices if 0 <= i < count]
        result: list[NVMLHandle] = []
        for i in wanted:
            try:
                result.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            except Exception as exc:
                logger.debug("Could not get handle for GPU %d: %s", i, exc)
        return result

    def driver_version(self) -> str | None:
        """Installed NVIDIA driver version string."""
        if not self._initialized:
            return None
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
            return version.decode() if isinstance(version, bytes) else str(version)
        except Exception:
            return None

    def device_name(self, handle: NVMLHandle) -> str | None:
        """Marketing name of the device (e.g. ``"NVIDIA GeForce GTX 1650"``)."""
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            return name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            return None

    def power_usage_w(self, handle: NVMLHandle) -> float | None:
        """Instantaneous board power draw in watts, if the sensor exists.

        Consumer cards such as the GTX 16xx series typically return an
        NVML "Not Supported" error here — callers must fall back to
        estimation.
        """
        try:
            return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception:
            return None

    def power_limit_w(self, handle: NVMLHandle) -> float | None:
        """Enforced board power limit in watts (a good TDP proxy)."""
        try:
            return float(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)) / 1000.0
        except Exception:
            pass
        try:
            return float(pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)) / 1000.0
        except Exception:
            return None

    def utilization(self, handle: NVMLHandle) -> tuple[float, float] | None:
        """(GPU %, memory-controller %) utilization over the last interval."""
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(rates.gpu), float(rates.memory)
        except Exception:
            return None

    def memory_info(self, handle: NVMLHandle) -> tuple[float, float] | None:
        """(used, total) device memory in MiB."""
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.used) / 2**20, float(info.total) / 2**20
        except Exception:
            return None

    def temperature_c(self, handle: NVMLHandle) -> float | None:
        """GPU core temperature in °C."""
        try:
            return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            return None
