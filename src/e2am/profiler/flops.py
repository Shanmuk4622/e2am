"""Static model profiling: parameters, MACs, and FLOPs.

MACs are counted with forward hooks on leaf modules during a single dummy
forward pass, so the counts reflect the *actual* execution path (dynamic
architectures included). The counter registry is open for extension: register
a counter for a new layer type with :func:`register_mac_counter` — no core
changes needed.

Honesty over false precision: parameters that live in modules E2AM cannot
count are reported in ``uncounted_modules`` together with a parameter
``coverage`` ratio, so users know when FLOPs are a lower bound.

FLOPs are reported using the common ``FLOPs = 2 × MACs`` convention
(one multiply + one accumulate).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from pydantic import BaseModel, Field
from torch import nn

from e2am.exceptions import ProfilerError
from e2am.utils.logging import get_logger

logger = get_logger("profiler.flops")

#: Counter signature: (module, input_tuple, output) -> MACs for this call.
MACCounter = Callable[[nn.Module, tuple, Any], int]

_MAC_COUNTERS: dict[type, MACCounter] = {}

#: Leaf modules that legitimately perform ~0 MACs (or negligibly few).
_ZERO_MAC_TYPES: tuple[type[nn.Module], ...] = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.GELU,
    nn.SiLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.Flatten,
    nn.Identity,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.Embedding,
)


def register_mac_counter(module_type: type, counter: MACCounter) -> None:
    """Register (or override) a MAC counter for a module type.

    Args:
        module_type: The ``nn.Module`` subclass to count.
        counter: Callable computing MACs for one forward call.
    """
    _MAC_COUNTERS[module_type] = counter


def _conv_macs(module: nn.Module, inputs: tuple, output: Any) -> int:
    kernel_ops = math.prod(module.kernel_size) * (module.in_channels // module.groups)
    return int(output.numel() * kernel_ops)


def _linear_macs(module: nn.Module, inputs: tuple, output: Any) -> int:
    return int(output.numel() * module.in_features)


for _conv_type in (nn.Conv1d, nn.Conv2d, nn.Conv3d):
    register_mac_counter(_conv_type, _conv_macs)
for _tconv_type in (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d):
    register_mac_counter(_tconv_type, _conv_macs)
register_mac_counter(nn.Linear, _linear_macs)


class ModelProfile(BaseModel):
    """Static complexity profile of a model for one input shape."""

    model_name: str = ""
    params: int = 0
    params_trainable: int = 0
    macs: int = Field(default=0, description="Multiply-accumulate operations per forward pass.")
    flops: int = Field(default=0, description="2 × MACs (multiply + add convention).")
    model_size_mb: float = Field(default=0.0, description="In-memory size of parameters + buffers.")
    input_shape: list[int] = Field(default_factory=list)
    device: str = ""
    coverage: float = Field(
        default=1.0,
        description="Fraction of parameters inside modules whose MACs were counted "
        "(1.0 = every parameterized module was profiled).",
    )
    uncounted_modules: list[str] = Field(
        default_factory=list,
        description="Parameterized leaf modules without a registered MAC counter.",
    )

    @property
    def gmacs(self) -> float:
        """MACs in billions."""
        return self.macs / 1e9

    @property
    def gflops(self) -> float:
        """FLOPs in billions."""
        return self.flops / 1e9


def _is_leaf(module: nn.Module) -> bool:
    return next(module.children(), None) is None


def profile_model(
    model: nn.Module,
    input_size: tuple[int, ...] | None = None,
    sample_input: torch.Tensor | tuple | None = None,
    device: torch.device | str | None = None,
) -> ModelProfile:
    """Profile parameters, MACs, and FLOPs of a model.

    Args:
        model: The model to profile.
        input_size: Full input shape *including* the batch dimension,
            e.g. ``(1, 3, 224, 224)``. Ignored when ``sample_input`` is given.
        sample_input: A ready-made input tensor (or tuple of tensors) for
            models with non-tensor or multi-input signatures.
        device: Device for the dummy forward pass. Defaults to the device of
            the model's first parameter.

    Returns:
        A :class:`ModelProfile` for one forward pass at the given shape.

    Raises:
        ProfilerError: If neither ``input_size`` nor ``sample_input`` is
            provided, or the forward pass fails.
    """
    if sample_input is None and input_size is None:
        raise ProfilerError("profile_model needs `input_size` or `sample_input`.")

    first_param = next(model.parameters(), None)
    if device is None:
        device = first_param.device if first_param is not None else torch.device("cpu")
    device = torch.device(device)

    if sample_input is None:
        assert input_size is not None
        dtype = first_param.dtype if first_param is not None else torch.float32
        sample_input = torch.randn(*input_size, device=device, dtype=dtype)
    inputs = sample_input if isinstance(sample_input, tuple) else (sample_input,)

    counted_macs = 0
    counted_param_ids: set[int] = set()
    uncounted: list[str] = []
    hooks = []

    def _make_hook(counter: MACCounter) -> Callable:
        def hook(module: nn.Module, inp: tuple, out: Any) -> None:
            nonlocal counted_macs
            counted_macs += counter(module, inp, out)

        return hook

    for name, module in model.named_modules():
        if not _is_leaf(module):
            continue
        counter = _MAC_COUNTERS.get(type(module))
        if counter is not None:
            hooks.append(module.register_forward_hook(_make_hook(counter)))
            counted_param_ids.update(id(p) for p in module.parameters())
        elif isinstance(module, _ZERO_MAC_TYPES):
            counted_param_ids.update(id(p) for p in module.parameters())
        elif any(True for _ in module.parameters()):
            uncounted.append(f"{name or '<root>'} ({type(module).__name__})")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(*inputs)
    except Exception as exc:
        raise ProfilerError(f"Forward pass failed while profiling: {exc}") from exc
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    del output

    params = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counted_params = sum(p.numel() for p in model.parameters() if id(p) in counted_param_ids)
    coverage = counted_params / params if params else 1.0
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_bytes += sum(b.numel() * b.element_size() for b in model.buffers())

    if uncounted:
        logger.warning(
            "MACs not counted for %d module(s) (%.0f%% of parameters covered): %s",
            len(uncounted),
            coverage * 100,
            ", ".join(uncounted[:5]) + ("..." if len(uncounted) > 5 else ""),
        )

    first = inputs[0]
    return ModelProfile(
        model_name=type(model).__name__,
        params=params,
        params_trainable=params_trainable,
        macs=counted_macs,
        flops=2 * counted_macs,
        model_size_mb=size_bytes / 2**20,
        input_shape=list(first.shape) if isinstance(first, torch.Tensor) else [],
        device=str(device),
        coverage=coverage,
        uncounted_modules=uncounted,
    )
