"""Torch-native classification metrics.

Computed from a confusion matrix built with ``bincount`` — no scikit-learn
dependency, works directly on the tensors a training loop already has, on
any device. Zero-division cases (a class never predicted / never present)
score 0 for that class, matching scikit-learn's ``zero_division=0``.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field

from e2am.exceptions import E2AMError


class ClassMetrics(BaseModel):
    """Precision/recall/F1 for one class."""

    label: int
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = Field(default=0, description="Number of true samples of this class.")


class ClassificationMetrics(BaseModel):
    """Aggregate classification quality metrics."""

    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    num_classes: int = 0
    num_samples: int = 0
    per_class: list[ClassMetrics] = Field(default_factory=list)


def confusion_matrix(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Confusion matrix with true classes as rows, predictions as columns."""
    indices = targets.long() * num_classes + predictions.long()
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
) -> ClassificationMetrics:
    """Compute accuracy, precision, recall, and F1 from predictions.

    Args:
        predictions: Either class labels of shape ``(N,)`` or
            logits/probabilities of shape ``(N, C)`` (argmax is applied).
        targets: True class labels of shape ``(N,)``.
        num_classes: Total classes. Inferred from the data when omitted.

    Returns:
        A :class:`ClassificationMetrics` with macro and support-weighted
        averages plus per-class detail.

    Raises:
        E2AMError: On empty or mismatched inputs.
    """
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu().long()
    if predictions.dim() == 2:
        if num_classes is None:
            num_classes = predictions.shape[1]
        predictions = predictions.argmax(dim=1)
    predictions = predictions.long()

    if predictions.numel() == 0:
        raise E2AMError("classification_metrics received empty predictions.")
    if predictions.shape != targets.shape:
        raise E2AMError(
            f"Shape mismatch: predictions {tuple(predictions.shape)} vs "
            f"targets {tuple(targets.shape)}."
        )
    if num_classes is None:
        num_classes = int(torch.max(torch.stack([predictions.max(), targets.max()])).item()) + 1

    cm = confusion_matrix(predictions, targets, num_classes).to(torch.float64)
    tp = cm.diag()
    support = cm.sum(dim=1)  # true samples per class
    predicted = cm.sum(dim=0)  # predicted samples per class
    total = int(support.sum().item())

    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(tp))

    weights = support / total
    per_class = [
        ClassMetrics(
            label=i,
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i in range(num_classes)
    ]

    return ClassificationMetrics(
        accuracy=float(tp.sum() / total),
        precision_macro=float(precision.mean()),
        recall_macro=float(recall.mean()),
        f1_macro=float(f1.mean()),
        precision_weighted=float((precision * weights).sum()),
        recall_weighted=float((recall * weights).sum()),
        f1_weighted=float((f1 * weights).sum()),
        num_classes=num_classes,
        num_samples=total,
        per_class=per_class,
    )
