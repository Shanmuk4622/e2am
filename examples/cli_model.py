"""The E2AM CLI script convention — use with `e2am train` / `e2am benchmark`.

    e2am train examples/cli_model.py --epochs 3
    e2am benchmark examples/cli_model.py --input-size 32,1,28,28

Required: get_model().  For training: get_loaders().
Optional: get_optimizer(model), get_loss().
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )


def get_loaders() -> tuple[DataLoader, DataLoader]:
    """Synthetic MNIST-shaped data so the example needs no downloads."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2048, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (2048,), generator=g)
    split = 1638
    return (
        DataLoader(TensorDataset(x[:split], y[:split]), batch_size=64, shuffle=True),
        DataLoader(TensorDataset(x[split:], y[split:]), batch_size=128),
    )


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=2e-3)
