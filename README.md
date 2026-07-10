<div align="center">

# ⚡ E2AM — Energy Efficient AI Models

**Automatic energy, carbon, and performance profiling for AI training — with almost zero code changes.**

[![CI](https://github.com/Shanmuk4622/e2am/actions/workflows/ci.yml/badge.svg)](https://github.com/Shanmuk4622/e2am/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
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
e2am hardware              # inspect detected hardware & energy capabilities
e2am train config.yaml     # run a training experiment from config
e2am benchmark model.py    # benchmark latency / throughput / energy
e2am report results/run1   # regenerate reports for a run
e2am compare run1 run2     # compare runs side by side
e2am dashboard             # launch the interactive dashboard
```

## Roadmap

- [x] Hardware detection with energy-capability probing
- [ ] Background energy/carbon/utilization monitoring (`monitor()`)
- [ ] FLOPs/MACs/latency profiler
- [ ] Drop-in `Trainer` with callbacks & plugins
- [ ] Automatic plots + HTML/PDF reports + leaderboard
- [ ] CLI (`train`, `benchmark`, `report`, `compare`, `dashboard`)
- [ ] Optimization engine (AMP, quantization, pruning, batch-size suggestions)
- [ ] Distributed training support
- [ ] Cloud dashboard & Hugging Face integration

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
