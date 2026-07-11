# E2AM architecture

## Layering

```
                    ┌───────────────────────────────┐
                    │        CLI (typer)            │  orchestrates, never reimplements
                    └──────────────┬────────────────┘
     ┌───────────────┬─────────────┴──┬───────────────────┐
┌────▼─────┐   ┌─────▼─────┐   ┌──────▼──────┐   ┌────────▼────────┐
│ Trainer  │   │ monitor() │   │  profiler   │   │ reports/viz     │
│ +callbcks│   │  (public) │   │ flops/lat/  │   │ html/md/pdf/csv │
└────┬─────┘   └─────┬─────┘   │   memory    │   │ dashboard/plots │
     │               │         └─────────────┘   └────────▲────────┘
     │        ┌──────▼────────────┐                       │
     │        │  MonitorSession   │── MonitorResult ──────┤
     └───────►│  (bg thread)      │                       │
              └──────┬────────────┘        TrainingResult─┘
       ┌─────────────┼──────────────┐
 ┌─────▼────┐  ┌─────▼────┐  ┌──────▼───┐
 │ Samplers │  │ Energy   │  │ Carbon   │
 │ cpu/ram/ │  │Integrator│  │Estimator │
 │ gpu(NVML)│  └──────────┘  └──────────┘
 └──────────┘
```

## Key decisions

| Decision | Rationale |
|---|---|
| `src/` layout, hatchling | prevents accidental repo-dir imports; modern packaging |
| torch is a peer dependency | users need hardware-matched builds; monitoring works without torch |
| Strategy-pattern samplers | per-device capability differences (NVML power sensor vs TDP×util fallback) stay local to one class |
| Trapezoidal incremental energy integration | correct with irregular tick spacing; O(1) memory; zero-order hold on missing readings |
| `measured` vs `estimated` flags everywhere | scientific honesty — consumers can see which numbers came from sensors |
| Callbacks as the only Trainer extension point | progress bars, early stopping, and future W&B/MLflow/Slack plugins share one lifecycle |
| `reports/common.build_sections` | HTML/Markdown/PDF render the same data; formats cannot drift |
| Package named `monitoring`, API named `monitor` | a submodule named `monitor` would shadow the public callable on `from e2am import monitor` |
| Leaderboard uses stdlib `csv` | the one artifact that must never fail should depend on nothing optional |
| Results are pydantic models | lossless JSON round-trip: every report can be regenerated from `metrics.json` alone |

## Energy model

- **GPU**: NVML live power sensor when the card exposes one; otherwise
  `board_power_limit × utilization` (flagged estimated).
- **CPU**: `TDP × utilization` (no portable OS power interface exists);
  TDP configurable via `monitor.cpu_tdp_w`.
- **RAM**: ~3 W per 8 GB *used* (CodeCarbon convention).
- **Carbon**: `kWh × grid intensity`, resolved as user override → country
  table (27 countries, Ember/IEA 2023) → world average 475 g/kWh.

## Green AI metrics

- **Green Score** `= 100 × accuracy × E_ref / (E_ref + E)`, `E_ref` = 0.1 kWh
  by default; comparable only under the same reference (recorded in results).
- **EAG (Energy-Accuracy Gradient)**: per-epoch `Δaccuracy(pct) / Δenergy(Wh)`
  — when it collapses toward zero, further energy is buying no accuracy.

## Threading model

One daemon thread per `MonitorSession`; all series appends happen under a
lock; `snapshot()` gives callbacks thread-safe running totals. `stop()` wakes
the thread promptly via `Event.wait`, takes a final sample, and tears down
samplers. Sampler and callback exceptions are logged, never propagated.
