"""Green AI metrics: efficiency, Green Score, and EAG.

These metrics relate model quality to its energy/carbon cost. They are pure
Python (no torch) so they can score any framework's runs.

**Green Score** (0–100, higher is better)::

    GS = 100 × accuracy × E_ref / (E_ref + E)

where ``E`` is the run's total energy and ``E_ref`` a reference energy
budget (default 0.1 kWh). The score is monotonic: it rises with accuracy and
falls as energy grows; a perfect model at negligible energy approaches 100,
and doubling energy beyond the reference roughly halves the energy factor.
Runs are only comparable under the same ``reference_kwh``, which is recorded
in the result.

**EAG (Energy-Accuracy Gradient)** is the discrete gradient of accuracy with
respect to cumulative energy across epochs, in *percentage points per Wh*.
A collapsing EAG means additional energy is no longer buying accuracy — the
scientifically interesting stopping signal for Green AI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from e2am.exceptions import E2AMError

#: Default Green Score reference energy in kWh.
DEFAULT_REFERENCE_KWH = 0.1


class GreenMetrics(BaseModel):
    """Efficiency metrics linking quality to energy/carbon cost."""

    energy_per_sample_j: float | None = None
    carbon_per_sample_mg: float | None = None
    samples_per_joule: float | None = None
    accuracy_per_joule: float | None = Field(
        default=None, description="Accuracy (0–1) divided by total energy in joules."
    )
    accuracy_per_kwh: float | None = None
    green_score: float | None = Field(
        default=None, description="Composite 0–100 score; see module docstring."
    )
    green_score_reference_kwh: float = DEFAULT_REFERENCE_KWH
    eag_pct_per_wh: float | None = Field(
        default=None,
        description="Final-epoch Energy-Accuracy Gradient in percentage points per Wh.",
    )


def green_score(
    accuracy: float, energy_kwh: float, reference_kwh: float = DEFAULT_REFERENCE_KWH
) -> float:
    """Compute the E2AM Green Score (see module docstring for the formula).

    Args:
        accuracy: Model quality in ``[0, 1]``.
        energy_kwh: Total energy consumed, in kWh (>= 0).
        reference_kwh: Reference energy budget anchoring the scale.

    Raises:
        E2AMError: On out-of-range inputs.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise E2AMError(f"accuracy must be in [0, 1], got {accuracy}.")
    if energy_kwh < 0:
        raise E2AMError(f"energy_kwh must be >= 0, got {energy_kwh}.")
    if reference_kwh <= 0:
        raise E2AMError(f"reference_kwh must be > 0, got {reference_kwh}.")
    return 100.0 * accuracy * reference_kwh / (reference_kwh + energy_kwh)


def energy_accuracy_gradient(
    accuracies: list[float], cumulative_energy_wh: list[float]
) -> list[float]:
    """Discrete gradient of accuracy w.r.t. cumulative energy per epoch.

    Args:
        accuracies: Accuracy (0–1) at the end of each epoch.
        cumulative_energy_wh: Total energy consumed *up to* each epoch (Wh),
            non-decreasing.

    Returns:
        Per-epoch gradients in percentage points per Wh; entry ``i`` is the
        gradient from epoch ``i`` to ``i+1`` (length ``len(accuracies) - 1``).

    Raises:
        E2AMError: On mismatched or too-short inputs.
    """
    if len(accuracies) != len(cumulative_energy_wh):
        raise E2AMError(
            "accuracies and cumulative_energy_wh must have equal length "
            f"({len(accuracies)} vs {len(cumulative_energy_wh)})."
        )
    if len(accuracies) < 2:
        raise E2AMError("EAG needs at least two epochs.")
    gradients: list[float] = []
    for i in range(1, len(accuracies)):
        d_energy = cumulative_energy_wh[i] - cumulative_energy_wh[i - 1]
        d_acc_pct = (accuracies[i] - accuracies[i - 1]) * 100.0
        gradients.append(d_acc_pct / d_energy if d_energy > 0 else 0.0)
    return gradients


def compute_green_metrics(
    energy_j: float,
    num_samples: int | None = None,
    accuracy: float | None = None,
    emissions_g: float | None = None,
    epoch_accuracies: list[float] | None = None,
    epoch_cumulative_energy_wh: list[float] | None = None,
    reference_kwh: float = DEFAULT_REFERENCE_KWH,
) -> GreenMetrics:
    """Compute every green metric derivable from the given measurements.

    Metrics whose inputs are missing are left as ``None`` rather than
    guessed.

    Args:
        energy_j: Total energy of the run in joules.
        num_samples: Total samples processed (for per-sample metrics).
        accuracy: Final model accuracy in ``[0, 1]``.
        emissions_g: Total carbon emissions in gCO2eq.
        epoch_accuracies: Per-epoch accuracies (for EAG).
        epoch_cumulative_energy_wh: Per-epoch cumulative energy (for EAG).
        reference_kwh: Green Score reference budget.

    Returns:
        A :class:`GreenMetrics` with all derivable fields populated.
    """
    energy_wh = energy_j / 3600.0
    energy_kwh = energy_wh / 1000.0
    metrics = GreenMetrics(green_score_reference_kwh=reference_kwh)

    if num_samples and num_samples > 0 and energy_j > 0:
        metrics.energy_per_sample_j = energy_j / num_samples
        metrics.samples_per_joule = num_samples / energy_j
        if emissions_g is not None:
            metrics.carbon_per_sample_mg = emissions_g * 1000.0 / num_samples

    if accuracy is not None:
        if energy_j > 0:
            metrics.accuracy_per_joule = accuracy / energy_j
            metrics.accuracy_per_kwh = accuracy / energy_kwh
        metrics.green_score = green_score(accuracy, energy_kwh, reference_kwh)

    if epoch_accuracies and epoch_cumulative_energy_wh and len(epoch_accuracies) >= 2:
        gradients = energy_accuracy_gradient(epoch_accuracies, epoch_cumulative_energy_wh)
        metrics.eag_pct_per_wh = gradients[-1]

    return metrics
