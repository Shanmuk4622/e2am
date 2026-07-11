"""FLOPs/MACs/params counting tests with hand-computed expectations."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from e2am.exceptions import ProfilerError  # noqa: E402
from e2am.profiler import ModelProfile, profile_model, register_mac_counter  # noqa: E402

pytestmark = pytest.mark.torch


def test_linear_macs_exact() -> None:
    # Linear 10 -> 5, batch 4: MACs = 4 * 5 * 10 = 200
    model = nn.Linear(10, 5)
    profile = profile_model(model, input_size=(4, 10))
    assert profile.macs == 200
    assert profile.flops == 400
    assert profile.params == 10 * 5 + 5
    assert profile.coverage == 1.0
    assert profile.uncounted_modules == []


def test_conv2d_macs_exact() -> None:
    # Conv2d 3->8, k=3, pad=1 on 16x16: out 8x16x16, kernel_ops = 3*3*3 = 27
    # MACs = 8*16*16 * 27 = 55296
    model = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    profile = profile_model(model, input_size=(1, 3, 16, 16))
    assert profile.macs == 8 * 16 * 16 * 27
    assert profile.params == 8 * 3 * 3 * 3 + 8


def test_small_cnn_composite() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),  # 4*8*8 * (1*3*3) = 2304
        nn.ReLU(),
        nn.MaxPool2d(2),  # 0 MACs
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 10),  # 10 * 64 = 640
    )
    profile = profile_model(model, input_size=(1, 1, 8, 8))
    assert profile.macs == 2304 + 640
    assert profile.params == sum(p.numel() for p in model.parameters())
    assert profile.params_trainable == profile.params
    assert profile.coverage == 1.0


def test_grouped_conv_macs() -> None:
    # groups=2: kernel_ops = (4/2)*3*3 = 18; out 8*10*10 => 8*10*10*18
    model = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=2)
    profile = profile_model(model, input_size=(1, 4, 10, 10))
    assert profile.macs == 8 * 10 * 10 * 18


def test_uncounted_module_reduces_coverage() -> None:
    class Weird(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(1000))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.weight.sum()

    model = nn.Sequential(nn.Linear(10, 10), Weird())
    profile = profile_model(model, input_size=(1, 10))
    assert profile.coverage < 1.0
    assert len(profile.uncounted_modules) == 1
    assert "Weird" in profile.uncounted_modules[0]


def test_custom_counter_registration() -> None:
    class Doubler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale

    register_mac_counter(Doubler, lambda module, inp, out: out.numel())
    profile = profile_model(Doubler(), input_size=(2, 7))
    assert profile.macs == 14
    assert profile.coverage == 1.0


def test_eval_and_training_mode_restored() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    model.train()
    profile_model(model, input_size=(1, 4))
    assert model.training  # restored

    model.eval()
    profile_model(model, input_size=(1, 4))
    assert not model.training


def test_missing_input_raises() -> None:
    with pytest.raises(ProfilerError, match="input_size"):
        profile_model(nn.Linear(2, 2))


def test_profile_serializes() -> None:
    profile = profile_model(nn.Linear(10, 5), input_size=(1, 10))
    restored = ModelProfile.model_validate_json(profile.model_dump_json())
    assert restored.macs == profile.macs
    assert restored.gflops == pytest.approx(profile.flops / 1e9)
