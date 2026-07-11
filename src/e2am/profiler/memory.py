"""Peak memory tracking around a block of code.

GPU peaks come from CUDA's allocator statistics (exact); host peaks are
approximated by RSS growth of the current process, which is what a user can
actually act on when sizing machines.
"""

from __future__ import annotations

import psutil
import torch
from pydantic import BaseModel

from e2am.utils.logging import get_logger

logger = get_logger("profiler.memory")


class MemoryUsage(BaseModel):
    """Memory observed while a :class:`MemoryTracker` block ran."""

    peak_gpu_mb: float | None = None
    cpu_rss_start_mb: float = 0.0
    cpu_rss_end_mb: float = 0.0

    @property
    def cpu_rss_delta_mb(self) -> float:
        """Host RSS growth over the tracked block."""
        return self.cpu_rss_end_mb - self.cpu_rss_start_mb


class MemoryTracker:
    """Context manager recording peak GPU memory and host RSS growth.

    Example:
        >>> with MemoryTracker() as mem:
        ...     train_one_epoch()
        >>> print(mem.usage.peak_gpu_mb)
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self._cuda = torch.cuda.is_available()
        self._device = torch.device(device) if device is not None else None
        self._process = psutil.Process()
        self.usage = MemoryUsage()

    def __enter__(self) -> MemoryTracker:
        self.usage.cpu_rss_start_mb = self._process.memory_info().rss / 2**20
        if self._cuda:
            try:
                torch.cuda.reset_peak_memory_stats(self._device)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("reset_peak_memory_stats failed: %s", exc)
                self._cuda = False
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.usage.cpu_rss_end_mb = self._process.memory_info().rss / 2**20
        if self._cuda:
            try:
                self.usage.peak_gpu_mb = torch.cuda.max_memory_allocated(self._device) / 2**20
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("max_memory_allocated failed: %s", exc)
