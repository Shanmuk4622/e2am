"""Carbon estimation tests."""

import pytest

from e2am.config.settings import WORLD_AVG_CARBON_INTENSITY, CarbonConfig
from e2am.monitoring.carbon import (
    CARBON_INTENSITY_BY_COUNTRY,
    CarbonEstimator,
)


def test_world_average_default() -> None:
    est = CarbonEstimator()
    assert est.intensity_g_per_kwh == WORLD_AVG_CARBON_INTENSITY
    assert est.intensity_source == "world_average"


def test_country_lookup() -> None:
    est = CarbonEstimator(CarbonConfig(country_iso_code="IND"))
    assert est.intensity_g_per_kwh == CARBON_INTENSITY_BY_COUNTRY["IND"]
    assert est.intensity_source == "country:IND"


def test_country_lookup_is_case_insensitive() -> None:
    est = CarbonEstimator(CarbonConfig(country_iso_code=" ind "))
    assert est.intensity_source == "country:IND"


def test_unknown_country_falls_back_to_world_average() -> None:
    est = CarbonEstimator(CarbonConfig(country_iso_code="XYZ"))
    assert est.intensity_g_per_kwh == WORLD_AVG_CARBON_INTENSITY
    assert est.intensity_source == "world_average"


def test_user_intensity_overrides_country() -> None:
    est = CarbonEstimator(CarbonConfig(country_iso_code="IND", carbon_intensity_g_per_kwh=100.0))
    assert est.intensity_g_per_kwh == 100.0
    assert est.intensity_source == "user"


def test_emission_arithmetic() -> None:
    est = CarbonEstimator(CarbonConfig(carbon_intensity_g_per_kwh=500.0))
    result = est.estimate(energy_kwh=2.0)
    assert result.emissions_g == pytest.approx(1000.0)
    assert result.intensity_g_per_kwh == 500.0
