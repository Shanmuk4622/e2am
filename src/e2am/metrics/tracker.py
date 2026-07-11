"""Lightweight time-series store for training metrics.

The :class:`MetricsTracker` is the single place a training loop (or the E2AM
``Trainer``) logs scalar metrics. It keeps insertion order, supports
auto-incrementing steps, and exports to plain dicts or a long-format pandas
``DataFrame`` for plotting and reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from e2am.exceptions import E2AMError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


class MetricsTracker:
    """Ordered scalar metric series keyed by name.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.log({"loss": 0.9, "acc": 0.4}, step=0)
        >>> tracker.log({"loss": 0.5, "acc": 0.7})  # step auto-increments to 1
        >>> tracker.series("loss")
        ([0, 1], [0.9, 0.5])
    """

    def __init__(self) -> None:
        self._steps: dict[str, list[int]] = {}
        self._values: dict[str, list[float]] = {}
        self._auto_step = 0

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record one or more metric values.

        Args:
            metrics: Mapping of metric name to scalar value.
            step: Explicit step index. When omitted, an internal counter is
                used and incremented.

        Raises:
            E2AMError: If a value is not a real number.
        """
        if step is None:
            step = self._auto_step
        self._auto_step = max(self._auto_step, step) + 1
        for name, value in metrics.items():
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise E2AMError(f"Metric {name!r} value {value!r} is not a real number.") from exc
            self._steps.setdefault(name, []).append(step)
            self._values.setdefault(name, []).append(value)

    @property
    def names(self) -> list[str]:
        """Metric names in first-logged order."""
        return list(self._values.keys())

    def series(self, name: str) -> tuple[list[int], list[float]]:
        """Return ``(steps, values)`` for one metric (copies)."""
        if name not in self._values:
            raise E2AMError(f"Unknown metric {name!r}; logged: {self.names}.")
        return list(self._steps[name]), list(self._values[name])

    def latest(self, name: str) -> float | None:
        """Most recent value of a metric, or ``None`` if never logged."""
        values = self._values.get(name)
        return values[-1] if values else None

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        """Export as ``{name: {"steps": [...], "values": [...]}}``."""
        return {
            name: {"steps": list(self._steps[name]), "values": list(self._values[name])}
            for name in self._values
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export as a long-format DataFrame with columns step/metric/value.

        Requires pandas, which E2AM does not install by default
        (``pip install pandas``).
        """
        import pandas as pd

        rows = [
            {"step": step, "metric": name, "value": value}
            for name in self._values
            for step, value in zip(self._steps[name], self._values[name], strict=True)
        ]
        return pd.DataFrame(rows, columns=["step", "metric", "value"])
