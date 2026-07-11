"""Shared visual style for E2AM plots.

Colors come from a CVD-validated categorical palette (worst adjacent-pair
color-vision-deficiency ΔE 24.2 in this order — the ordering is the safety
mechanism, so assign slots in order, never cycle arbitrary hues). Chrome
(ink/grid/surface) is deliberately recessive so the data carries the figure.
"""

from __future__ import annotations

#: Categorical series colors in validated order (light surface).
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FIGSIZE = (8.0, 4.5)
DPI = 150
LINE_WIDTH = 2.0


def assign_channel_colors(channels: list[str]) -> dict[str, str]:
    """Map channel names to palette slots, stably within a run.

    Channels are sorted before assignment so the same device always gets the
    same color across every plot of a run (color follows the entity).
    """
    palette = {}
    for i, name in enumerate(sorted(channels)):
        palette[name] = SERIES_COLORS[i % len(SERIES_COLORS)]
    return palette
