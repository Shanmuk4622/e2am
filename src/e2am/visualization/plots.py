"""Publication-quality PNG plots for monitoring and training results.

Implementation notes:

* Figures are built with matplotlib's object-oriented API and the Agg canvas
  directly — never ``pyplot`` — so E2AM cannot disturb the user's interactive
  backend or leak global figure state from a background save.
* Every generator returns the list of files it actually wrote; plots whose
  data is missing are skipped silently (a CPU-only run simply has no
  ``gpu_usage.png``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from e2am.monitoring.result import MonitorResult
from e2am.trainer.result import TrainingResult
from e2am.utils.logging import get_logger
from e2am.visualization.style import (
    BASELINE,
    DPI,
    FIGSIZE,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    LINE_WIDTH,
    SURFACE,
    assign_channel_colors,
)

logger = get_logger("visualization.plots")

Series = dict[str, tuple[list[float], list[float]]]


def _new_figure(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Any]:
    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    return fig, ax


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


def _plot_lines(
    series: Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    percent_axis: bool = False,
    integer_x: bool = False,
) -> Path | None:
    """Line plot of one or more named series; legend only when >= 2."""
    series = {name: (x, y) for name, (x, y) in series.items() if len(y) > 0}
    if not series:
        return None
    fig, ax = _new_figure(title, xlabel, ylabel)
    colors = assign_channel_colors(list(series))
    for name, (x, y) in series.items():
        ax.plot(x, y, label=name, color=colors[name], linewidth=LINE_WIDTH, solid_capstyle="round")
    if percent_axis:
        ax.set_ylim(0, 105)
    if integer_x:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(series) >= 2:
        legend = ax.legend(frameon=False, fontsize=8, loc="best")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
    return _save(fig, path)


def _clean_channel(x: list[float], y: list[float | None]) -> tuple[list[float], list[float]]:
    pairs = [(xi, yi) for xi, yi in zip(x, y, strict=False) if yi is not None]
    if not pairs:
        return [], []
    xs, ys = zip(*pairs, strict=True)
    return list(xs), list(ys)


def _cumulative_energy_wh(x: list[float], y: list[float]) -> list[float]:
    """Trapezoidal cumulative energy (Wh) of a power series (W over s)."""
    total = 0.0
    out = [0.0]
    for i in range(1, len(x)):
        total += (y[i] + y[i - 1]) / 2.0 * (x[i] - x[i - 1]) / 3600.0
        out.append(total)
    return out


def generate_monitor_plots(monitor: MonitorResult, out_dir: str | Path) -> list[Path]:
    """Write power/energy/usage/memory/carbon plots for a monitored run.

    Returns:
        Paths of the PNG files actually written (data-less plots skipped).
    """
    out_dir = Path(out_dir)
    ts = monitor.timeseries
    t = ts.timestamps_s
    written: list[Path] = []

    power: Series = {}
    for name, values in ts.power_w.items():
        x, y = _clean_channel(t, values)
        if y:
            power[name] = (x, y)
    path = _plot_lines(power, "Power draw", "time (s)", "power (W)", out_dir / "power.png")
    if path:
        written.append(path)

    energy: Series = {
        name: (x, _cumulative_energy_wh(x, y)) for name, (x, y) in power.items() if len(x) > 1
    }
    path = _plot_lines(
        energy, "Cumulative energy", "time (s)", "energy (Wh)", out_dir / "energy.png"
    )
    if path:
        written.append(path)

    gpu_util: Series = {}
    cpu_util: Series = {}
    for name, values in ts.utilization_pct.items():
        x, y = _clean_channel(t, values)
        if not y:
            continue
        if name.startswith("gpu"):
            gpu_util[name] = (x, y)
        elif name == "cpu":
            cpu_util[name] = (x, y)
    path = _plot_lines(
        gpu_util,
        "GPU utilization",
        "time (s)",
        "utilization (%)",
        out_dir / "gpu_usage.png",
        percent_axis=True,
    )
    if path:
        written.append(path)
    path = _plot_lines(
        cpu_util,
        "CPU utilization",
        "time (s)",
        "utilization (%)",
        out_dir / "cpu_usage.png",
        percent_axis=True,
    )
    if path:
        written.append(path)

    memory: Series = {}
    for name, values in ts.memory_used_mb.items():
        x, y = _clean_channel(t, values)
        if y:
            memory[name] = (x, y)
    path = _plot_lines(memory, "Memory used", "time (s)", "memory (MB)", out_dir / "memory.png")
    if path:
        written.append(path)

    if power:
        total_x = max(power.values(), key=lambda xy: len(xy[0]))[0]
        summed: list[float] = []
        for i, _ in enumerate(total_x):
            summed.append(sum(y[i] for _, y in power.values() if len(y) > i))
        cumulative_g = [
            wh / 1000.0 * monitor.carbon.intensity_g_per_kwh
            for wh in _cumulative_energy_wh(total_x, summed)
        ]
        path = _plot_lines(
            {"emissions": (total_x, cumulative_g)},
            f"Cumulative carbon (grid intensity {monitor.carbon.intensity_g_per_kwh:.0f} g/kWh)",
            "time (s)",
            "gCO2eq",
            out_dir / "carbon.png",
        )
        if path:
            written.append(path)

    logger.debug("Wrote %d monitor plot(s) to %s", len(written), out_dir)
    return written


def generate_training_plots(result: TrainingResult, out_dir: str | Path) -> list[Path]:
    """Write loss/accuracy/latency/throughput plots for a training run."""
    out_dir = Path(out_dir)
    written: list[Path] = []

    def _epoch_series(name: str) -> tuple[list[float], list[float]]:
        entry = result.history.get(name)
        if not entry:
            return [], []
        return [s + 1 for s in entry["steps"]], list(entry["values"])

    loss: Series = {}
    for key, label in (("train_loss", "train"), ("val_loss", "validation")):
        x, y = _epoch_series(key)
        if y:
            loss[label] = (x, y)
    path = _plot_lines(loss, "Loss", "epoch", "loss", out_dir / "loss.png", integer_x=True)
    if path:
        written.append(path)

    accuracy: Series = {}
    x, y = _epoch_series("val_accuracy")
    if y:
        accuracy["validation"] = (x, y)
    path = _plot_lines(
        accuracy, "Accuracy", "epoch", "accuracy", out_dir / "accuracy.png", integer_x=True
    )
    if path:
        written.append(path)

    x, y = _epoch_series("epoch_time_s")
    path = _plot_lines(
        {"epoch time": (x, y)},
        "Epoch time",
        "epoch",
        "seconds",
        out_dir / "latency.png",
        integer_x=True,
    )
    if path:
        written.append(path)

    if y and result.epochs_completed:
        samples_per_epoch = result.samples_processed / result.epochs_completed
        throughput = [samples_per_epoch / t if t > 0 else 0.0 for t in y]
        path = _plot_lines(
            {"throughput": (x, throughput)},
            "Training throughput",
            "epoch",
            "samples / s",
            out_dir / "throughput.png",
            integer_x=True,
        )
        if path:
            written.append(path)

    if result.monitor is not None:
        written.extend(generate_monitor_plots(result.monitor, out_dir))
    logger.debug("Wrote %d training plot(s) to %s", len(written), out_dir)
    return written


def generate_plots(result: TrainingResult | MonitorResult, out_dir: str | Path) -> list[Path]:
    """Write every plot derivable from a result object."""
    if isinstance(result, TrainingResult):
        return generate_training_plots(result, out_dir)
    return generate_monitor_plots(result, out_dir)
