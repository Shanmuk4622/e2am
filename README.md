<div align="center">

# ⚡ E2AM — Energy Efficient AI Models

**Automatic energy, carbon, and performance profiling for AI training — with almost zero code changes.**

[![CI](https://github.com/Shanmuk4622/e2am/actions/workflows/ci.yml/badge.svg)](https://github.com/Shanmuk4622/e2am/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

E2AM makes AI model training automatically measurable in terms of **energy efficiency**,
**carbon emissions**, **computational cost**, and **model performance**. Think of it as
*Weights & Biases + CodeCarbon + PyTorch Profiler* for **Green AI** — in one lightweight toolkit
for researchers, ML engineers, universities, and companies.

## Quickstart

Wrap any training code:

```python
from e2am import monitor

with monitor(project="ResNet50"):
    train()
```

Or use the drop-in trainer:

```python
from e2am import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
)
trainer.fit()
```

Everything else happens automatically: energy, carbon, utilization, FLOPs, timing,
accuracy metrics, plots, and reports land in `results/`.

## Installation

```bash
pip install e2am
```

> **Note:** PyTorch is not installed automatically — install the build matching your
> hardware from [pytorch.org](https://pytorch.org/get-started/locally/). Monitoring
> works even without PyTorch.

Optional extras:

```bash
pip install "e2am[carbon]"     # CodeCarbon-backed carbon tracking
pip install "e2am[pdf]"        # PDF reports (reportlab)
pip install "e2am[dashboard]"  # interactive dashboards (plotly)
pip install "e2am[all]"        # everything
```

## What E2AM measures

| Category    | Metrics                                                                 |
| ----------- | ----------------------------------------------------------------------- |
| Energy      | GPU / CPU / RAM energy (Wh), power draw over time, energy per sample    |
| Carbon      | CO₂eq emissions, carbon per sample, region-aware carbon intensity       |
| Compute     | FLOPs, MACs, parameters, GPU/CPU utilization, peak memory               |
| Time        | Training / epoch / batch time, latency, throughput                      |
| Quality     | Accuracy, precision, recall, F1                                          |
| Green AI    | Accuracy per joule, **Green Score**, **EAG** (Energy-Accuracy Gradient) |

## Automatic outputs

```
results/<run_name>/
    accuracy.png   loss.png      energy.png     power.png
    gpu_usage.png  cpu_usage.png memory.png     carbon.png
    latency.png    throughput.png
    report.html    report.pdf
    metrics.json   leaderboard.csv
    config.yaml    README.md
```

## Architecture

```
e2am/
    trainer/         # drop-in Trainer with callback lifecycle
    monitor/         # background samplers: GPU (NVML), CPU, RAM, carbon
    profiler/        # FLOPs / MACs / params, latency, memory
    metrics/         # classification + Green AI metrics
    benchmark/       # model/hardware benchmarking
    reports/         # HTML / PDF / JSON / Markdown report generation
    visualization/   # publication-quality plots
    dashboard/       # interactive dashboard
    plugins/         # W&B, MLflow, TensorBoard, Slack, Discord, HF
    cli/             # `e2am` command line
    config/          # typed, YAML-loadable configuration
    utils/           # hardware detection, logging, timing
```

Design principles: SOLID, clean architecture, graceful degradation (no GPU? no NVML power
sensor? no problem — E2AM falls back to utilization-based estimation), and zero required
code changes to your training loop.

## CLI

```bash
e2am hardware                                   # detected hardware & energy capabilities
e2am train model.py --config config.yaml        # train with full telemetry
e2am benchmark model.py --input-size 8,3,224,224  # FLOPs, latency, J/inference
e2am report results/run1                        # regenerate reports for a run
e2am compare results/run1 results/run2          # side-by-side comparison
e2am optimize results/run1                      # efficiency suggestions + Wh savings
e2am dashboard                                  # local HTML dashboard of all runs
```

`model.py` follows a tiny convention (see [examples/cli_model.py](examples/cli_model.py)):
define `get_model()` and, for training, `get_loaders()`; optionally
`get_optimizer(model)` and `get_loss()`.

## Plugins

Every plugin is a Trainer callback — pass any combination:

```python
from e2am import Trainer
from e2am.plugins import WandbPlugin, MLflowPlugin, TensorBoardPlugin, SlackPlugin

trainer = Trainer(
    model=model, optimizer=optimizer, train_loader=train_loader,
    callbacks=[
        WandbPlugin(entity="my-team"),          # pip install wandb
        MLflowPlugin(experiment_name="green"),  # pip install mlflow
        TensorBoardPlugin(),                    # pip install tensorboard
        SlackPlugin("https://hooks.slack.com/services/..."),  # no extra deps
    ],
)
```

Slack/Discord notify on completion **and on failure** — including the final
energy, carbon, and Green Score. Missing packages fail fast at construction
with the pip command; network errors during training are logged, never raised.
Write your own by subclassing `e2am.trainer.Callback`.

## Hugging Face

Already using `transformers.Trainer`? One callback adds full E2AM telemetry:

```python
from transformers import Trainer
from e2am.integrations import E2AMCallback

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
    callbacks=[E2AMCallback(project="bert-finetune")],
)
trainer.train()   # energy, carbon, green metrics + full E2AM reports in results/
```

## Roadmap

- [x] Hardware detection with energy-capability probing
- [x] Background energy/carbon/utilization monitoring (`monitor()`)
- [x] FLOPs/MACs/latency/peak-memory profiler
- [x] Drop-in `Trainer` with callback lifecycle (AMP, grad accumulation, early stopping)
- [x] Green AI metrics: energy/carbon per sample, accuracy/joule, Green Score, EAG
- [x] Automatic plots + HTML/Markdown/PDF reports + leaderboard
- [x] CLI (`hardware`, `train`, `benchmark`, `report`, `compare`, `dashboard`)
- [x] Local HTML dashboard across runs
- [x] Plugin integrations (W&B, MLflow, TensorBoard, Slack/Discord)
- [x] Optimization engine (`e2am optimize`: AMP, batch size, torch.compile, checkpointing, quantization, wasted-epoch detection with measured Wh savings)
- [x] Hugging Face `transformers` integration (`E2AMCallback`)
- [ ] Distributed training support
- [ ] Cloud dashboard

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md). Development setup:

```bash
git clone https://github.com/Shanmuk4622/e2am.git
cd e2am
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
