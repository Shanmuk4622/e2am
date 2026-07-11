"""Peak-memory tracker tests."""

import pytest

torch = pytest.importorskip("torch")

from e2am.profiler import MemoryTracker  # noqa: E402

pytestmark = pytest.mark.torch


def test_cpu_rss_recorded() -> None:
    with MemoryTracker() as mem:
        _ = [bytearray(1024) for _ in range(100)]
    assert mem.usage.cpu_rss_start_mb > 0
    assert mem.usage.cpu_rss_end_mb > 0
    assert isinstance(mem.usage.cpu_rss_delta_mb, float)


@pytest.mark.gpu
def test_gpu_peak_tracks_allocation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    with MemoryTracker() as mem:
        x = torch.zeros(256, 1024, 1024, device="cuda")  # 1 GiB fp32
        del x
        torch.cuda.empty_cache()
    assert mem.usage.peak_gpu_mb is not None
    assert mem.usage.peak_gpu_mb >= 1024  # at least the 1 GiB we allocated
