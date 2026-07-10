"""NVML wrapper tests — exercise graceful degradation on every machine."""

import pytest

from e2am.utils.nvml import NVML


def test_nvml_lifecycle_never_raises() -> None:
    nvml = NVML()
    nvml.init()
    nvml.shutdown()
    nvml.shutdown()  # double shutdown is safe


def test_uninitialized_queries_are_safe() -> None:
    nvml = NVML()
    assert nvml.device_count() == 0
    assert nvml.handles() == []
    assert nvml.driver_version() is None


def test_context_manager() -> None:
    with NVML() as nvml:
        assert isinstance(nvml.available, bool)


@pytest.mark.gpu
def test_gpu_queries_return_sane_values() -> None:
    with NVML() as nvml:
        if not nvml.available or nvml.device_count() == 0:
            pytest.skip("No NVML-visible GPU on this machine")
        handle = nvml.handles()[0]
        assert nvml.device_name(handle)
        mem = nvml.memory_info(handle)
        assert mem is not None and mem[1] > 0
        util = nvml.utilization(handle)
        if util is not None:
            assert 0 <= util[0] <= 100
        power = nvml.power_usage_w(handle)
        if power is not None:
            assert 0 < power < 1500
