# Changelog

All notable changes to E2AM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-11

First public release. 🌱

### Added

**Monitoring**
- `with monitor(project=...)` context manager / decorator — zero-code-change
  energy, carbon, and utilization tracking for any code block.
- Background `MonitorSession` with per-device samplers: GPU (NVML live power
  sensor, TDP × utilization fallback for cards without one), CPU
  (TDP × utilization), RAM (~3 W per 8 GB used); live `snapshot()` totals.
- Trapezoidal power→energy integration robust to irregular intervals and
  missing readings; measured-vs-estimated flags on every device.
- Region-aware carbon estimation (27-country intensity table, user override,
  world-average fallback).

**Profiling & metrics**
- `profile_model`: hook-based MACs/FLOPs/params with an extensible counter
  registry and honest parameter-coverage reporting.
- `benchmark_latency`: warmed-up, CUDA-synchronized latency (mean/p50/p95)
  and throughput. `MemoryTracker`: exact CUDA peak + host RSS delta.
- Torch-native classification metrics (accuracy, macro/weighted P/R/F1).
- Green AI metrics: energy/carbon per sample, accuracy-per-joule,
  Green Score (`100·acc·E_ref/(E_ref+E)`), EAG (energy-accuracy gradient).

**Training**
- Drop-in `Trainer` with AMP, gradient accumulation, clipping, LR scheduler
  support, callback lifecycle, `EarlyStopping`, and automatic integration of
  monitoring + profiling + green metrics.

**Outputs**
- Automatic artifacts per run: 10 plots, self-contained `report.html`,
  `README.md`, optional `report.pdf`, `metrics.json`, `summary.yaml`,
  `config.yaml`, cross-run `leaderboard.csv`, local `dashboard.html`.

**CLI**
- `e2am hardware | train | benchmark | report | compare | optimize | dashboard`.
- `benchmark` reports energy per inference (joules) via live monitoring.
- `optimize`: rule-based efficiency suggestions with quantified Wh savings
  (wasted-epoch detection, AMP, batch size, torch.compile, checkpointing,
  quantization).

**Integrations**
- Plugins (all Trainer callbacks): Weights & Biases, MLflow, TensorBoard,
  Slack, Discord (webhooks are stdlib-only).
- Hugging Face: `E2AMCallback` for `transformers.Trainer`.

### Notes
- Python ≥ 3.10. PyTorch is a peer dependency (install the build matching
  your hardware); monitoring works without it.
