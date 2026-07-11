"""Automatic energy, carbon, and utilization monitoring."""

from __future__ import annotations

from e2am.monitoring.api import monitor
from e2am.monitoring.carbon import CarbonEstimator, CarbonResult
from e2am.monitoring.energy import EnergyIntegrator, joules_to_kwh, joules_to_wh
from e2am.monitoring.result import (
    DeviceEnergy,
    MonitorResult,
    TimeSeriesData,
    UtilizationStats,
)
from e2am.monitoring.samplers import (
    CPUSampler,
    GPUSampler,
    RAMSampler,
    Sample,
    Sampler,
    create_samplers,
)
from e2am.monitoring.session import MonitorSession

__all__ = [
    "CPUSampler",
    "CarbonEstimator",
    "CarbonResult",
    "DeviceEnergy",
    "EnergyIntegrator",
    "GPUSampler",
    "MonitorResult",
    "MonitorSession",
    "RAMSampler",
    "Sample",
    "Sampler",
    "TimeSeriesData",
    "UtilizationStats",
    "create_samplers",
    "joules_to_kwh",
    "joules_to_wh",
    "monitor",
]
