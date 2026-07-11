"""Green AI metric tests — formulas verified by hand."""

import pytest

from e2am.exceptions import E2AMError
from e2am.metrics import (
    compute_green_metrics,
    energy_accuracy_gradient,
    green_score,
)


def test_green_score_formula() -> None:
    # GS = 100 * acc * ref / (ref + E)
    assert green_score(1.0, 0.0) == pytest.approx(100.0)
    assert green_score(0.9, 0.1, reference_kwh=0.1) == pytest.approx(45.0)
    assert green_score(0.5, 0.3, reference_kwh=0.1) == pytest.approx(12.5)


def test_green_score_monotonicity() -> None:
    assert green_score(0.9, 0.1) > green_score(0.8, 0.1)  # better accuracy wins
    assert green_score(0.9, 0.1) > green_score(0.9, 0.2)  # less energy wins


def test_green_score_validation() -> None:
    with pytest.raises(E2AMError):
        green_score(1.5, 0.1)
    with pytest.raises(E2AMError):
        green_score(0.5, -1.0)
    with pytest.raises(E2AMError):
        green_score(0.5, 0.1, reference_kwh=0.0)


def test_eag_hand_computed() -> None:
    # epoch accs: 0.5 -> 0.7 -> 0.8; cumulative Wh: 10 -> 20 -> 40
    # grad1 = (0.2*100)/10 = 2.0 pct/Wh; grad2 = (0.1*100)/20 = 0.5 pct/Wh
    grads = energy_accuracy_gradient([0.5, 0.7, 0.8], [10.0, 20.0, 40.0])
    assert grads == pytest.approx([2.0, 0.5])


def test_eag_zero_energy_delta_is_zero() -> None:
    grads = energy_accuracy_gradient([0.5, 0.9], [10.0, 10.0])
    assert grads == [0.0]


def test_eag_validation() -> None:
    with pytest.raises(E2AMError, match="equal length"):
        energy_accuracy_gradient([0.5], [1.0, 2.0])
    with pytest.raises(E2AMError, match="two epochs"):
        energy_accuracy_gradient([0.5], [1.0])


def test_compute_green_metrics_full() -> None:
    # 7200 J = 2 Wh = 0.002 kWh; 1000 samples; acc 0.8; 1 g CO2
    m = compute_green_metrics(
        energy_j=7200.0,
        num_samples=1000,
        accuracy=0.8,
        emissions_g=1.0,
        epoch_accuracies=[0.5, 0.8],
        epoch_cumulative_energy_wh=[1.0, 2.0],
        reference_kwh=0.1,
    )
    assert m.energy_per_sample_j == pytest.approx(7.2)
    assert m.samples_per_joule == pytest.approx(1000 / 7200)
    assert m.carbon_per_sample_mg == pytest.approx(1.0)
    assert m.accuracy_per_joule == pytest.approx(0.8 / 7200)
    assert m.accuracy_per_kwh == pytest.approx(0.8 / 0.002)
    assert m.green_score == pytest.approx(100 * 0.8 * 0.1 / 0.102)
    assert m.eag_pct_per_wh == pytest.approx(30.0)  # (0.3*100)/1


def test_compute_green_metrics_partial_inputs() -> None:
    m = compute_green_metrics(energy_j=100.0)
    assert m.energy_per_sample_j is None
    assert m.green_score is None
    assert m.eag_pct_per_wh is None
