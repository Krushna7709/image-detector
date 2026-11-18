"""Shared metrics: detection, calibration and mask overlap."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:  # scikit-learn ships with the project requirements, but keep this import soft
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional dependency path
    roc_auc_score = None


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def expected_calibration_error(
    y_true: Sequence[float], probabilities: Sequence[float], n_bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=np.float32).ravel()
    p = np.clip(np.asarray(probabilities, dtype=np.float32).ravel(), 0.0, 1.0)
    if y.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        members = bins == index
        if not members.any():
            continue
        confidence = float(p[members].mean())
        accuracy = float(y[members].mean())
        error += (members.sum() / y.size) * abs(confidence - accuracy)
    return float(error)


def brier_score(y_true: Sequence[float], probabilities: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.float32).ravel()
    p = np.clip(np.asarray(probabilities, dtype=np.float32).ravel(), 0.0, 1.0)
    if y.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def classification_metrics(
    y_true: Sequence[float], probabilities: Sequence[float], threshold: float = 0.5
) -> dict[str, object]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.clip(np.asarray(probabilities, dtype=np.float32).ravel(), 0.0, 1.0)
    if y.size == 0:
        return {"n": 0}
    preds = (p >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    auc = None
    if roc_auc_score is not None and len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
    return {
        "n": int(y.size),
        "positives": int(y.sum()),
        "negatives": int(y.size - y.sum()),
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(np.mean(preds == y)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "brier": round(brier_score(y, p), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "real": {"real": tn, "ai": fp},
            "ai": {"real": fn, "ai": tp},
        },
    }


def best_threshold(
    y_true: Sequence[float],
    probabilities: Sequence[float],
    metric: str = "balanced_accuracy",
) -> float:
    """Pick a decision threshold from validation probabilities.

    Ties are resolved toward 0.5 so the threshold generalises rather than
    chasing a single validation ordering.
    """

    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.clip(np.asarray(probabilities, dtype=np.float32).ravel(), 0.0, 1.0)
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.5
    if metric not in {"balanced_accuracy", "f1", "accuracy"}:
        raise ValueError(f"Unknown threshold metric '{metric}'.")

    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    total = int(y.size)
    total_positive = int(y.sum())
    cumulative_positive = np.concatenate(([0], np.cumsum(sorted_y)))
    unique = np.unique(p)
    candidates = np.unique(
        np.clip(
            np.concatenate(([0.01, 0.5], (unique[:-1] + unique[1:]) / 2.0, [0.99])),
            0.01,
            0.99,
        )
    )

    best_score = -1.0
    best_value = 0.5
    for candidate in candidates:
        below = int(np.searchsorted(sorted_p, candidate, side="left"))
        tp = total_positive - int(cumulative_positive[below])
        fp = (total - below) - tp
        fn = total_positive - tp
        tn = below - (total_positive - tp)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        if metric == "f1":
            score = _safe_divide(2 * precision * recall, precision + recall)
        elif metric == "accuracy":
            score = _safe_divide(tp + tn, total)
        else:
            score = (recall + specificity) / 2.0
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and abs(candidate - 0.5) < abs(best_value - 0.5)
        ):
            best_score = score
            best_value = float(candidate)
    return best_value


def mask_metrics(
    predicted: np.ndarray, target: np.ndarray
) -> dict[str, float] | None:
    pred = np.asarray(predicted, dtype=bool).ravel()
    truth = np.asarray(target, dtype=bool).ravel()
    if pred.size != truth.size:
        raise ValueError("Predicted and target masks have different shapes.")
    intersection = float(np.sum(pred & truth))
    union = float(np.sum(pred | truth))
    if union == 0.0:
        return None
    precision = _safe_divide(intersection, float(np.sum(pred)))
    recall = _safe_divide(intersection, float(np.sum(truth)))
    return {
        "iou": _safe_divide(intersection, union),
        "dice": _safe_divide(2.0 * intersection, float(np.sum(pred) + np.sum(truth))),
        "precision": precision,
        "recall": recall,
    }


def aggregate_mask_metrics(items: Iterable[dict[str, float]]) -> dict[str, float]:
    values = list(items)
    if not values:
        return {"images": 0}
    keys = ("iou", "dice", "precision", "recall")
    summary: dict[str, float] = {"images": float(len(values))}
    for key in keys:
        summary[key] = float(np.mean([value[key] for value in values]))
    return summary


def confusion_matrix_text(confusion: dict[str, dict[str, int]]) -> str:
    header = "            predicted_real  predicted_ai"
    rows = [
        f"true_real   {confusion['real']['real']:>14d}  {confusion['real']['ai']:>12d}",
        f"true_ai     {confusion['ai']['real']:>14d}  {confusion['ai']['ai']:>12d}",
    ]
    return "\n".join([header, *rows])
