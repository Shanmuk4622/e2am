"""Shared test fixtures: synthetic results that mimic a real short run."""

from datetime import datetime, timedelta, timezone

import pytest

from e2am.monitoring.carbon import CarbonResult
from e2am.monitoring.result import (
    DeviceEnergy,
    MonitorResult,
    TimeSeriesData,
    UtilizationStats,
)


@pytest.fixture
def monitor_result() -> MonitorResult:
    """A hand-built MonitorResult with 5 samples across cpu/ram/gpu0."""
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    started = datetime(2026, 7, 11, 10, 0, 0, tzinfo=timezone.utc)
    return MonitorResult(
        project="proj",
        run_name="fake-run",
        started_at=started,
        ended_at=started + timedelta(seconds=4),
        duration_s=4.0,
        sampling_interval_s=1.0,
        num_samples=5,
        devices=[
            DeviceEnergy(name="cpu", energy_j=80.0, avg_power_w=20.0, max_power_w=30.0),
            DeviceEnergy(name="ram", energy_j=12.0, avg_power_w=3.0, max_power_w=3.0),
            DeviceEnergy(
                name="gpu0",
                energy_j=200.0,
                avg_power_w=50.0,
                max_power_w=70.0,
                power_estimated=False,
            ),
        ],
        utilization={
            "cpu": UtilizationStats(avg_pct=25.0, max_pct=60.0),
            "ram": UtilizationStats(avg_pct=80.0, max_pct=85.0),
            "gpu0": UtilizationStats(avg_pct=90.0, max_pct=99.0),
        },
        peak_memory_mb={"ram": 6200.0, "gpu0": 1800.0},
        carbon=CarbonResult(
            emissions_g=0.04, intensity_g_per_kwh=475.0, intensity_source="world_average"
        ),
        timeseries=TimeSeriesData(
            timestamps_s=t,
            power_w={
                "cpu": [10.0, 20.0, 30.0, 25.0, 15.0],
                "ram": [3.0, 3.0, 3.0, 3.0, 3.0],
                "gpu0": [20.0, 60.0, 70.0, 65.0, 35.0],
            },
            utilization_pct={
                "cpu": [10.0, 30.0, 60.0, 40.0, 20.0],
                "ram": [78.0, 80.0, 82.0, 85.0, 80.0],
                "gpu0": [50.0, 95.0, 99.0, 97.0, 60.0],
            },
            memory_used_mb={
                "cpu": [None, None, None, None, None],
                "ram": [6000.0, 6100.0, 6200.0, 6150.0, 6050.0],
                "gpu0": [500.0, 1700.0, 1800.0, 1750.0, 600.0],
            },
            temperature_c={
                "cpu": [None] * 5,
                "ram": [None] * 5,
                "gpu0": [55.0, 62.0, 68.0, 66.0, 58.0],
            },
        ),
    )
