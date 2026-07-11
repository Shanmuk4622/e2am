"""Shared section-building logic for all report formats.

Every report format (HTML, Markdown, PDF) renders the same information; this
module computes it once as ordered ``(label, value)`` sections so the formats
cannot drift apart.
"""

from __future__ import annotations

from e2am.monitoring.result import MonitorResult
from e2am.trainer.result import TrainingResult

AnyResult = TrainingResult | MonitorResult

Section = tuple[str, list[tuple[str, str]]]


def _fmt(value: float | None, spec: str = ".4g", suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:{spec}}{suffix}"


def _monitor_of(result: AnyResult) -> MonitorResult | None:
    if isinstance(result, MonitorResult):
        return result
    return result.monitor


def build_sections(result: AnyResult) -> list[Section]:
    """Compute every report section for a result, in display order."""
    sections: list[Section] = []
    monitor = _monitor_of(result)

    summary: list[tuple[str, str]] = [
        ("Project", result.project),
        ("Run", result.run_name),
        ("Status", result.status),
    ]
    if isinstance(result, TrainingResult):
        summary += [
            ("Epochs", f"{result.epochs_completed}/{result.epochs_requested}"),
            ("Samples processed", f"{result.samples_processed:,}"),
            ("Avg epoch time", _fmt(result.avg_epoch_time_s, ".3g", " s")),
            ("Avg batch time", _fmt(result.avg_batch_time_ms, ".3g", " ms")),
            ("Throughput", _fmt(result.train_throughput_samples_per_s, ".4g", " samples/s")),
        ]
    if monitor is not None:
        summary += [
            ("Started", monitor.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")),
            ("Duration", _fmt(monitor.duration_s, ".4g", " s")),
            ("Total energy", _fmt(monitor.total_energy_wh, ".4g", " Wh")),
            (
                "Carbon",
                f"{monitor.carbon.emissions_g:.4g} gCO2eq "
                f"({monitor.carbon.intensity_g_per_kwh:.0f} g/kWh, "
                f"{monitor.carbon.intensity_source})",
            ),
        ]
    sections.append(("Experiment summary", summary))

    if monitor is not None:
        system = monitor.system
        hardware = [
            ("Host", f"{system.hostname} ({system.os} {system.os_version})"),
            ("Python", system.python_version),
            (
                "CPU",
                f"{system.cpu.model} "
                f"({system.cpu.physical_cores or '?'}c/{system.cpu.logical_cores or '?'}t, "
                f"TDP {system.cpu.tdp_w:.0f} W)",
            ),
            ("RAM", f"{system.ram.total_gb:.1f} GB"),
        ]
        for gpu in system.gpus:
            sensor = "power sensor" if gpu.supports_power_reading else "estimated power"
            hardware.append(
                (
                    f"GPU {gpu.index}",
                    f"{gpu.name} ({_fmt(gpu.total_memory_mb, '.0f', ' MB')}, "
                    f"driver {gpu.driver_version or '?'}, {sensor})",
                )
            )
        if system.torch_version:
            cuda = f", CUDA {system.cuda_version}" if system.cuda_available else ", CPU-only"
            hardware.append(("PyTorch", f"{system.torch_version}{cuda}"))
        sections.append(("Hardware", hardware))

        devices = []
        for device in monitor.devices:
            tag = " (estimated)" if device.power_estimated else " (measured)"
            devices.append(
                (
                    device.name,
                    f"{device.energy_wh:.4g} Wh · avg {device.avg_power_w:.1f} W · "
                    f"max {device.max_power_w:.1f} W{tag}",
                )
            )
        for name, stats in sorted(monitor.utilization.items()):
            devices.append(
                (f"{name} utilization", f"avg {stats.avg_pct:.1f} % · max {stats.max_pct:.1f} %")
            )
        for name, peak in sorted(monitor.peak_memory_mb.items()):
            devices.append((f"{name} peak memory", f"{peak:.0f} MB"))
        sections.append(("Energy & utilization", devices))

    if isinstance(result, TrainingResult):
        quality = [
            ("Best val accuracy", _fmt(result.best_val_accuracy)),
            (
                "Best epoch",
                _fmt(None if result.best_epoch is None else result.best_epoch + 1, ".0f"),
            ),
            ("Final train loss", _fmt(result.final_train_loss)),
            ("Final val loss", _fmt(result.final_val_loss)),
            ("Final val accuracy", _fmt(result.final_val_accuracy)),
            ("Final val F1 (macro)", _fmt(result.final_val_f1_macro)),
        ]
        sections.append(("Model quality", quality))

        if result.profile:
            profile = result.profile
            sections.append(
                (
                    "Model profile",
                    [
                        ("Model", str(profile.get("model_name", "?"))),
                        ("Parameters", f"{profile.get('params', 0):,}"),
                        ("Trainable", f"{profile.get('params_trainable', 0):,}"),
                        ("MACs / forward", f"{profile.get('macs', 0):,}"),
                        ("FLOPs / forward", f"{profile.get('flops', 0):,}"),
                        ("Model size", _fmt(profile.get("model_size_mb"), ".4g", " MB")),
                        ("MAC coverage", _fmt(100 * profile.get("coverage", 1.0), ".3g", " %")),
                    ],
                )
            )

        if result.green is not None:
            green = result.green
            sections.append(
                (
                    "Green AI metrics",
                    [
                        ("Energy / sample", _fmt(green.energy_per_sample_j, ".4g", " J")),
                        ("Carbon / sample", _fmt(green.carbon_per_sample_mg, ".4g", " mg")),
                        ("Samples / joule", _fmt(green.samples_per_joule)),
                        ("Accuracy / kWh", _fmt(green.accuracy_per_kwh, ".4g")),
                        (
                            "Green Score",
                            _fmt(green.green_score, ".4g")
                            + f" (ref {green.green_score_reference_kwh} kWh)",
                        ),
                        ("EAG (final)", _fmt(green.eag_pct_per_wh, ".4g", " %-pts/Wh")),
                    ],
                )
            )

    return sections
