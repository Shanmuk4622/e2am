"""Sampler tests — CPU/RAM run anywhere; GPU tests need real hardware."""

import pytest

from e2am.config.settings import MonitorConfig
from e2am.monitoring.samplers import CPUSampler, GPUSampler, RAMSampler, create_samplers


def test_cpu_sampler_gives_sane_values() -> None:
    sampler = CPUSampler(tdp_w=50.0)
    assert sampler.setup()
    sample = sampler.sample()
    assert sample.utilization_pct is not None
    assert 0.0 <= sample.utilization_pct <= 100.0
    assert sample.power_w is not None
    assert 0.0 <= sample.power_w <= 50.0
    assert sampler.estimates_power


def test_ram_sampler_gives_sane_values() -> None:
    sampler = RAMSampler()
    assert sampler.setup()
    sample = sampler.sample()
    assert sample.power_w is not None and sample.power_w > 0
    assert sample.memory_used_mb is not None and sample.memory_used_mb > 0
    assert sample.memory_total_mb is not None
    assert sample.memory_used_mb <= sample.memory_total_mb
    assert sampler.estimates_power


def test_gpu_sampler_for_missing_index_is_unavailable() -> None:
    sampler = GPUSampler(index=99)
    assert sampler.setup() is False
    sampler.teardown()


def test_create_samplers_always_includes_cpu_and_ram() -> None:
    samplers = create_samplers(MonitorConfig())
    names = {s.name for s in samplers}
    assert {"cpu", "ram"}.issubset(names)
    for s in samplers:
        s.teardown()


@pytest.mark.gpu
def test_gpu_sampler_on_real_hardware() -> None:
    sampler = GPUSampler(index=0)
    if not sampler.setup():
        pytest.skip("No GPU 0 visible via NVML")
    try:
        sample = sampler.sample()
        assert sample.power_w is not None
        assert 0.0 < sample.power_w < 1500.0
        assert sample.memory_used_mb is not None
        assert sampler.tdp_w > 0
    finally:
        sampler.teardown()
