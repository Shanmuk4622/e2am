# Contributing to E2AM

Thanks for helping make AI training measurably greener! 🌱

## Development setup

```bash
git clone https://github.com/Shanmuk4622/e2am.git
cd e2am
python -m venv .venv && source .venv/bin/activate   # or your conda env
pip install -e ".[dev,pdf]"
```

PyTorch is a peer dependency — install the build matching your hardware from
[pytorch.org](https://pytorch.org/get-started/locally/).

## Quality gate

Every PR must pass the same gate CI runs:

```bash
ruff check src tests
black --check src tests
isort --check-only src tests
mypy src/e2am
pytest -m "not gpu"        # GPU tests run only on machines with an NVIDIA GPU
```

On a GPU machine, also run the full suite: `pytest`.

## Design rules of the codebase

- **Graceful degradation** — no GPU, no power sensor, no torch: E2AM must
  still work, estimating (and *labeling* the estimate) where it cannot measure.
- **Never crash user training** — samplers, callbacks, and report generation
  are failure-isolated; a monitoring bug must not kill an 8-hour run.
- **Honesty over false precision** — anything estimated is flagged as
  estimated in results, reports, and plots.
- **One implementation** — the CLI orchestrates the Python API; it never
  reimplements it.
- **Torch stays optional at import time** — `import e2am` and monitoring work
  without PyTorch; torch-backed modules are imported lazily.

## Style

Python ≥ 3.10, full type hints, Google-style docstrings, `ruff`/`black`/`isort`
formatting (line length 100). Tests use plain `pytest` with markers `gpu`,
`torch`, `slow`. New metrics or samplers need hand-computed expected values in
their tests, not just "doesn't crash".

## Adding a sampler / metric / report

- New device sampler: subclass `e2am.monitoring.samplers.Sampler`, register in
  `create_samplers`, never raise from `sample()`.
- New MAC counter: `e2am.profiler.register_mac_counter(YourModule, fn)`.
- New report format: consume `e2am.reports.common.build_sections` so every
  format stays in sync.

## Reporting issues

Include your OS, Python, torch version, GPU/driver, and the output of
`e2am hardware --json`.
