"""Drop-in Trainer with full Green AI telemetry on a synthetic dataset.

Runs on CPU or GPU, no downloads needed:
    python examples/quickstart_trainer.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from e2am import Trainer
from e2am.trainer import EarlyStopping


def make_data(n: int = 4000) -> tuple[DataLoader, DataLoader]:
    """Two Gaussian blobs — learnable in a couple of epochs."""
    g = torch.Generator().manual_seed(42)
    x = torch.cat(
        [torch.randn(n // 2, 16, generator=g) - 1, torch.randn(n // 2, 16, generator=g) + 1]
    )
    y = torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)])
    split = int(0.8 * n)
    train = TensorDataset(x[:split], y[:split])
    val = TensorDataset(x[split:], y[split:])
    return (DataLoader(train, batch_size=64, shuffle=True), DataLoader(val, batch_size=128))


model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
train_loader, val_loader = make_data()

trainer = Trainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    train_loader=train_loader,
    val_loader=val_loader,
    project="quickstart",
    run_name="trainer-demo",
    epochs=10,
    mixed_precision=True,  # auto-disabled on CPU
    callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
)
result = trainer.fit()

print(f"\nAccuracy     : {result.final_val_accuracy:.3f}")
print(f"Energy       : {result.total_energy_wh:.4f} Wh")
print(f"Green Score  : {result.green.green_score:.1f}/100")
print(f"Report       : results/{result.run_name}/report.html")
