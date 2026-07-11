"""Energy integrator tests — exact trapezoidal arithmetic."""

import pytest

from e2am.monitoring.energy import EnergyIntegrator, joules_to_kwh, joules_to_wh


def test_unit_conversions() -> None:
    assert joules_to_wh(3600.0) == pytest.approx(1.0)
    assert joules_to_kwh(3_600_000.0) == pytest.approx(1.0)


def test_constant_power() -> None:
    integ = EnergyIntegrator()
    integ.add(0.0, 100.0)
    assert integ.add(2.0, 100.0) == pytest.approx(200.0)
    assert integ.energy_wh == pytest.approx(200.0 / 3600.0)


def test_trapezoid_with_ramping_power() -> None:
    integ = EnergyIntegrator()
    integ.add(0.0, 0.0)
    # ramp 0 -> 100 W over 10 s: trapezoid = (0+100)/2 * 10 = 500 J
    assert integ.add(10.0, 100.0) == pytest.approx(500.0)


def test_irregular_intervals() -> None:
    integ = EnergyIntegrator()
    integ.add(0.0, 50.0)
    integ.add(1.0, 50.0)
    integ.add(4.5, 50.0)  # long gap still integrates correctly
    assert integ.energy_j == pytest.approx(50.0 * 4.5)


def test_none_reading_holds_last_power() -> None:
    integ = EnergyIntegrator()
    integ.add(0.0, 100.0)
    integ.add(1.0, None)  # hold 100 W for 1 s
    assert integ.energy_j == pytest.approx(100.0)
    integ.add(2.0, 100.0)  # last known power still 100 W
    assert integ.energy_j == pytest.approx(200.0)


def test_leading_none_contributes_nothing() -> None:
    integ = EnergyIntegrator()
    integ.add(0.0, None)
    integ.add(1.0, None)
    assert integ.energy_j == 0.0
    integ.add(2.0, 100.0)  # first real reading; no prior power to integrate
    assert integ.energy_j == pytest.approx(100.0)  # power*dt fallback


def test_out_of_order_points_ignored() -> None:
    integ = EnergyIntegrator()
    integ.add(5.0, 100.0)
    before = integ.add(6.0, 100.0)
    assert integ.add(2.0, 500.0) == before
