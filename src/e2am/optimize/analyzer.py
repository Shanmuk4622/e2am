"""Rule-based optimization analysis of a finished training run.

Each rule is a pure function ``TrainingResult -> Suggestion | None`` that
inspects the run's *measurements* (utilization, per-epoch energy/accuracy,
memory pressure, configuration) and only fires when the data supports it.
Estimates are honest about their basis: savings derived from this run's own
numbers are reported in Wh; literature-typical ranges are reported as
percentages.
"""

from __future__ import annotations

from collections.abc import Callable

from e2am.optimize.suggestions import PRIORITY_ORDER, Suggestion
from e2am.trainer.result import TrainingResult
from e2am.utils.logging import get_logger

logger = get_logger("optimize.analyzer")

#: Accuracy within this absolute distance of the best counts as "converged".
CONVERGENCE_EPSILON = 0.002

Rule = Callable[[TrainingResult], "Suggestion | None"]


def _gpu_energy_wh(result: TrainingResult) -> float:
    return result.monitor.gpu_energy_wh if result.monitor else 0.0


def _avg_util(result: TrainingResult, channel: str) -> float | None:
    if result.monitor is None or channel not in result.monitor.utilization:
        return None
    return result.monitor.utilization[channel].avg_pct


def rule_mixed_precision(result: TrainingResult) -> Suggestion | None:
    """AMP unused on a CUDA run — the single easiest energy win."""
    if result.mixed_precision or not result.device.startswith("cuda"):
        return None
    gpu_wh = _gpu_energy_wh(result)
    return Suggestion(
        id="mixed-precision",
        title="Enable automatic mixed precision (AMP)",
        priority="high",
        rationale=(
            f"This run trained in full precision on {result.device}. AMP typically "
            "cuts GPU time and energy by 15–40 % on tensor-core GPUs with no "
            "accuracy loss for most models."
        ),
        action="Trainer(..., mixed_precision=True)",
        estimated_savings_pct=(15.0, 40.0),
        estimated_savings_wh=round(gpu_wh * 0.25, 6) if gpu_wh else None,
    )


def rule_wasted_epochs(result: TrainingResult) -> Suggestion | None:
    """Energy spent after validation accuracy stopped improving (EAG ≈ 0)."""
    accuracy = result.history.get("val_accuracy")
    energy = result.history.get("cumulative_energy_wh")
    if not accuracy or not energy or len(accuracy["values"]) < 3:
        return None
    acc_values = accuracy["values"]
    energy_values = energy["values"]
    if len(acc_values) != len(energy_values):
        return None
    best = max(acc_values)
    converged_epoch = next(
        i for i, acc in enumerate(acc_values) if acc >= best - CONVERGENCE_EPSILON
    )
    if converged_epoch >= len(acc_values) - 1:
        return None  # kept improving to the very end
    wasted_wh = energy_values[-1] - energy_values[converged_epoch]
    if wasted_wh <= 0:
        return None
    wasted_epochs = len(acc_values) - 1 - converged_epoch
    total_wh = energy_values[-1]
    pct = 100.0 * wasted_wh / total_wh if total_wh > 0 else 0.0
    return Suggestion(
        id="early-stopping",
        title=f"Stop earlier: the last {wasted_epochs} epoch(s) bought no accuracy",
        priority="high",
        rationale=(
            f"Validation accuracy reached within {CONVERGENCE_EPSILON:.1%} of its best "
            f"({best:.4f}) at epoch {converged_epoch + 1}, but training continued for "
            f"{wasted_epochs} more epoch(s), consuming {wasted_wh:.4g} Wh "
            f"({pct:.0f} % of the run) with an energy-accuracy gradient near zero."
        ),
        action=(
            "from e2am.trainer import EarlyStopping\n"
            'Trainer(..., callbacks=[EarlyStopping(monitor="val_accuracy", '
            'mode="max", patience=2)])'
        ),
        estimated_savings_wh=round(wasted_wh, 6),
    )


def rule_gpu_underutilized(result: TrainingResult) -> Suggestion | None:
    """GPU mostly idle — batches too small or the input pipeline can't feed it."""
    if not result.device.startswith("cuda"):
        return None
    gpu_util = _avg_util(result, "gpu0")
    if gpu_util is None or gpu_util >= 50.0:
        return None
    cpu_util = _avg_util(result, "cpu")
    feeding_problem = cpu_util is not None and cpu_util > 80.0
    if feeding_problem:
        rationale = (
            f"Average GPU utilization was only {gpu_util:.0f} % while CPU utilization "
            f"was {cpu_util:.0f} % — the input pipeline is starving the GPU."
        )
        action = "DataLoader(..., num_workers=4, pin_memory=True, persistent_workers=True)"
    else:
        rationale = (
            f"Average GPU utilization was only {gpu_util:.0f} %. The GPU idles between "
            "kernels; larger batches amortize fixed per-step overhead."
        )
        action = "Increase batch_size (e.g. 2–4x) until GPU memory or accuracy limits are hit."
    return Suggestion(
        id="gpu-underutilized",
        title="Raise GPU utilization (larger batches / faster input pipeline)",
        priority="high",
        rationale=rationale,
        action=action,
        estimated_savings_pct=(10.0, 30.0),
    )


def rule_torch_compile(result: TrainingResult) -> Suggestion | None:
    """torch.compile speeds up steady-state training on PyTorch 2.x."""
    version = result.monitor.system.torch_version if result.monitor else None
    if not version or not version.split(".")[0].isdigit() or int(version.split(".")[0]) < 2:
        return None
    if not result.device.startswith("cuda"):
        return None
    return Suggestion(
        id="torch-compile",
        title="Compile the model with torch.compile",
        priority="medium",
        rationale=(
            f"PyTorch {version} supports torch.compile; kernel fusion typically "
            "yields 5–20 % faster steps (and proportionally less energy) after "
            "a one-time compilation warmup. Best for runs with many steps."
        ),
        action="model = torch.compile(model)",
        estimated_savings_pct=(5.0, 20.0),
    )


def rule_memory_pressure(result: TrainingResult) -> Suggestion | None:
    """Peak GPU memory near capacity constrains batch size."""
    if result.monitor is None or not result.monitor.system.gpus:
        return None
    total = result.monitor.system.gpus[0].total_memory_mb
    peak = result.monitor.peak_memory_mb.get("gpu0")
    if not total or peak is None or peak / total < 0.85:
        return None
    return Suggestion(
        id="gradient-checkpointing",
        title="Use gradient checkpointing to relieve memory pressure",
        priority="medium",
        rationale=(
            f"Peak GPU memory was {peak:.0f} MB of {total:.0f} MB "
            f"({100 * peak / total:.0f} %). Checkpointing trades ~20–30 % more "
            "compute for large activation-memory savings, unlocking bigger "
            "batches (often a net efficiency win)."
        ),
        action="model.gradient_checkpointing_enable()  # HF models\n"
        "# or torch.utils.checkpoint.checkpoint(...) around large blocks",
        estimated_savings_pct=None,
    )


def rule_quantized_inference(result: TrainingResult) -> Suggestion | None:
    """Large models can be quantized for deployment."""
    params = (result.profile or {}).get("params", 0)
    if not isinstance(params, int) or params < 1_000_000:
        return None
    return Suggestion(
        id="quantized-inference",
        title="Quantize the model for inference deployment",
        priority="low",
        rationale=(
            f"The model has {params:,} parameters. INT8 dynamic quantization "
            "typically shrinks it ~4x and cuts inference energy per sample "
            "substantially, with small accuracy cost."
        ),
        action=(
            "quantized = torch.ao.quantization.quantize_dynamic(\n"
            "    model, {torch.nn.Linear}, dtype=torch.qint8)"
        ),
        estimated_savings_pct=(30.0, 60.0),
    )


#: All rules, evaluated in order.
RULES: tuple[Rule, ...] = (
    rule_mixed_precision,
    rule_wasted_epochs,
    rule_gpu_underutilized,
    rule_torch_compile,
    rule_memory_pressure,
    rule_quantized_inference,
)


def analyze(result: TrainingResult) -> list[Suggestion]:
    """Run every rule against a finished run.

    Args:
        result: A completed (or stopped) training result.

    Returns:
        Suggestions sorted by priority (high first), then rule order.
    """
    suggestions: list[Suggestion] = []
    for rule in RULES:
        try:
            suggestion = rule(result)
        except Exception as exc:  # a broken rule must not block the others
            logger.warning("Optimization rule %s failed: %s", rule.__name__, exc)
            continue
        if suggestion is not None:
            suggestions.append(suggestion)
    return sorted(suggestions, key=lambda s: PRIORITY_ORDER.get(s.priority, 9))
