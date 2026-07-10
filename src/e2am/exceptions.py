"""Exception hierarchy for E2AM.

All E2AM exceptions inherit from :class:`E2AMError`, so callers can catch a
single base class. Modules raise the most specific subclass available.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "E2AMError",
    "HardwareError",
    "MonitorError",
    "ProfilerError",
    "ReportError",
    "TrainerError",
]


class E2AMError(Exception):
    """Base class for all E2AM errors."""


class ConfigError(E2AMError):
    """Raised when configuration is invalid or cannot be loaded."""


class HardwareError(E2AMError):
    """Raised when hardware detection or access fails unexpectedly."""


class MonitorError(E2AMError):
    """Raised when the monitoring session cannot start or record data."""


class ProfilerError(E2AMError):
    """Raised when model profiling (FLOPs, latency, memory) fails."""


class TrainerError(E2AMError):
    """Raised when the training loop is misconfigured or fails."""


class ReportError(E2AMError):
    """Raised when report generation fails."""
