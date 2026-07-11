"""Incremental power → energy integration.

Energy is the time integral of power. The integrator applies the trapezoidal
rule incrementally so it works with irregular sampling intervals and never
needs to hold the full series in memory. Missing readings (``None``) fall
back to a zero-order hold on the last known power, which is far more accurate
over a long run than dropping the interval entirely.
"""

from __future__ import annotations

JOULES_PER_WH = 3600.0
WH_PER_KWH = 1000.0


def joules_to_wh(joules: float) -> float:
    """Convert joules to watt-hours."""
    return joules / JOULES_PER_WH


def joules_to_kwh(joules: float) -> float:
    """Convert joules to kilowatt-hours."""
    return joules / (JOULES_PER_WH * WH_PER_KWH)


class EnergyIntegrator:
    """Accumulates energy (J) from timestamped power readings (W).

    Example:
        >>> integ = EnergyIntegrator()
        >>> integ.add(0.0, 100.0)
        0.0
        >>> integ.add(2.0, 100.0)  # 100 W for 2 s
        200.0
    """

    def __init__(self) -> None:
        self._last_t: float | None = None
        self._last_p: float | None = None
        self._energy_j = 0.0

    @property
    def energy_j(self) -> float:
        """Total accumulated energy in joules."""
        return self._energy_j

    @property
    def energy_wh(self) -> float:
        """Total accumulated energy in watt-hours."""
        return joules_to_wh(self._energy_j)

    def add(self, t_s: float, power_w: float | None) -> float:
        """Feed one reading and return the updated total energy in joules.

        Args:
            t_s: Monotonic timestamp of the reading in seconds. Must be
                non-decreasing across calls; out-of-order points are ignored.
            power_w: Power at that instant, or ``None`` if the reading failed
                (the last known power is held for the elapsed interval).

        Returns:
            Cumulative energy in joules after this reading.
        """
        if self._last_t is not None and t_s < self._last_t:
            return self._energy_j

        if self._last_t is not None:
            dt = t_s - self._last_t
            if power_w is not None and self._last_p is not None:
                self._energy_j += (power_w + self._last_p) / 2.0 * dt
            elif power_w is not None:
                self._energy_j += power_w * dt
            elif self._last_p is not None:
                self._energy_j += self._last_p * dt

        self._last_t = t_s
        if power_w is not None:
            self._last_p = power_w
        return self._energy_j
