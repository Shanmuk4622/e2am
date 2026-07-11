"""Classification metric tests with hand-computed expectations."""

import pytest

torch = pytest.importorskip("torch")

from e2am.exceptions import E2AMError  # noqa: E402
from e2am.metrics import classification_metrics, confusion_matrix  # noqa: E402

pytestmark = pytest.mark.torch


def test_perfect_predictions() -> None:
    preds = torch.tensor([0, 1, 2, 0, 1, 2])
    m = classification_metrics(preds, preds.clone())
    assert m.accuracy == 1.0
    assert m.f1_macro == 1.0
    assert m.precision_weighted == 1.0
    assert m.num_classes == 3
    assert m.num_samples == 6


def test_hand_computed_binary_case() -> None:
    # targets: [1,1,1,0,0]; preds: [1,1,0,0,1]
    # class1: tp=2, fp=1, fn=1 -> P=2/3, R=2/3, F1=2/3
    # class0: tp=1, fp=1, fn=1 -> P=1/2, R=1/2, F1=1/2
    targets = torch.tensor([1, 1, 1, 0, 0])
    preds = torch.tensor([1, 1, 0, 0, 1])
    m = classification_metrics(preds, targets)
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.precision_macro == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m.recall_macro == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m.f1_macro == pytest.approx((2 / 3 + 1 / 2) / 2)
    # weighted by support (class0: 2, class1: 3)
    assert m.f1_weighted == pytest.approx((2 * 0.5 + 3 * (2 / 3)) / 5)
    c1 = m.per_class[1]
    assert c1.precision == pytest.approx(2 / 3)
    assert c1.support == 3


def test_logits_are_argmaxed() -> None:
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = torch.tensor([1, 0, 1])
    m = classification_metrics(logits, targets)
    assert m.accuracy == 1.0
    assert m.num_classes == 2


def test_never_predicted_class_scores_zero() -> None:
    # class 2 exists in targets but is never predicted
    targets = torch.tensor([0, 1, 2])
    preds = torch.tensor([0, 1, 0])
    m = classification_metrics(preds, targets, num_classes=3)
    c2 = m.per_class[2]
    assert c2.precision == 0.0
    assert c2.recall == 0.0
    assert c2.f1 == 0.0


def test_confusion_matrix_layout() -> None:
    # rows = true class, cols = predicted class
    targets = torch.tensor([0, 0, 1])
    preds = torch.tensor([0, 1, 1])
    cm = confusion_matrix(preds, targets, num_classes=2)
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_empty_and_mismatched_inputs_raise() -> None:
    with pytest.raises(E2AMError, match="empty"):
        classification_metrics(torch.tensor([]), torch.tensor([]))
    with pytest.raises(E2AMError, match="mismatch"):
        classification_metrics(torch.tensor([0, 1]), torch.tensor([0]))


def test_gpu_tensors_accepted_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    preds = torch.tensor([0, 1], device="cuda")
    targets = torch.tensor([0, 1], device="cuda")
    assert classification_metrics(preds, targets).accuracy == 1.0
