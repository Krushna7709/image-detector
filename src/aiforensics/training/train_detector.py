"""Train the CLIP + linear real-vs-AI detector.

Consumes the ``.npz`` produced by ``aiforensics.training.clip_features`` and
writes the artifact read by ``LinearDetector``::

    python -m aiforensics.training.clip_features --manifest data/index.csv --out artifacts/clip.npz
    python -m aiforensics.training.train_detector --features artifacts/clip.npz --out models/detector.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aiforensics.config import settings as default_settings
from aiforensics.training.calibration import CalibratedClassifier, fit_calibrator
from aiforensics.training.clip_features import load_features
from aiforensics.training.metrics import (
    best_threshold,
    classification_metrics,
    confusion_matrix_text,
)


def _select(data: dict[str, np.ndarray], split: str):
    mask = data["splits"] == split
    return data["features"][mask], data["labels"][mask], data["generators"][mask]


def _positive_probabilities(estimator, features: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.predict_proba(features))[:, 1]


def _generator_report(
    labels: np.ndarray, probabilities: np.ndarray, generators: np.ndarray, threshold: float
) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for generator in sorted(set(generators.tolist())):
        mask = generators == generator
        if not mask.any():
            continue
        report[generator] = {
            "n": int(mask.sum()),
            "mean_probability": round(float(probabilities[mask].mean()), 4),
            "positive_rate": round(float((probabilities[mask] >= threshold).mean()), 4),
            "ai_label_rate": round(float(labels[mask].mean()), 4),
        }
    return report


def train_detector(
    features_path: Path,
    out_path: Path,
    C: float = 1.0,
    seed: int = 13,
    calibrate: bool = True,
    threshold_metric: str = "balanced_accuracy",
) -> dict[str, object]:
    data = load_features(features_path)
    x_train, y_train, generators_train = _select(data, "train")
    x_val, y_val, _ = _select(data, "val")
    x_test, y_test, generators_test = _select(data, "test")
    if x_train.size == 0:
        raise RuntimeError("No training split found in the feature file.")
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("The training split needs both real and AI images.")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    notes: list[str] = []
    raw_threshold = 0.5
    if len(np.unique(y_val)) == 2:
        raw_probabilities = _positive_probabilities(pipeline, x_val)
        raw_threshold = best_threshold(y_val, raw_probabilities, threshold_metric)
        threshold_source = "val"
    elif len(np.unique(y_test)) == 2:
        raw_threshold = best_threshold(y_test, _positive_probabilities(pipeline, x_test), threshold_metric)
        threshold_source = "test"
        notes.append("Validation split lacked both classes; threshold tuned on test and metrics are optimistic.")
    else:
        threshold_source = "default"
        notes.append("No labelled validation split; using the 0.5 threshold.")

    estimator = pipeline
    calibrated = False
    threshold = raw_threshold
    calibration_dict = None
    if calibrate and len(np.unique(y_val)) == 2:
        raw_val = _positive_probabilities(pipeline, x_val)
        calibrator = fit_calibrator(raw_val, y_val)
        if calibrator is not None:
            estimator = CalibratedClassifier(pipeline, calibrator)
            calibrated = True
            calibration_dict = calibrator.to_dict()
            calibrated_val = _positive_probabilities(estimator, x_val)
            threshold = best_threshold(y_val, calibrated_val, threshold_metric)
        else:
            notes.append("Platt calibration was not fitted; raw probabilities retained.")
    elif calibrate:
        notes.append("Platt calibration skipped because the validation split has one class.")

    val_metrics = None
    if x_val.size:
        val_metrics = classification_metrics(
            y_val, _positive_probabilities(estimator, x_val), threshold
        )
    test_metrics = None
    if x_test.size:
        test_metrics = classification_metrics(
            y_test, _positive_probabilities(estimator, x_test), threshold
        )
        test_metrics["per_generator"] = _generator_report(
            y_test, _positive_probabilities(estimator, x_test), generators_test, threshold
        )

    trained_on = sorted(set(generators_train[y_train == 1].tolist()))
    artifact = {
        "pipeline": estimator,
        "threshold": float(threshold),
        "raw_threshold": float(raw_threshold),
        "trained_on": trained_on,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "calibrated": calibrated,
        "calibration": calibration_dict,
        "threshold_source": threshold_source,
        "threshold_metric": threshold_metric,
        "feature": "clip",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "C": C,
        "notes": notes,
    }
    destination = Path(out_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)

    print(f"Saved detector to {destination} (calibrated={calibrated}, threshold={threshold:.3f})")
    print(f"Training generators: {', '.join(trained_on) or 'none'}")
    for split_name, metrics in (("val", val_metrics), ("test", test_metrics)):
        if not metrics:
            continue
        print(
            f"{split_name}: n={metrics['n']} accuracy={metrics['accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} auc={metrics['roc_auc']} ece={metrics['ece']:.3f}"
        )
        print(confusion_matrix_text(metrics["confusion"]))
        for generator, stats in metrics.get("per_generator", {}).items():
            print(
                f"  {generator:20s} n={stats['n']:5d} "
                f"mean_p={stats['mean_probability']:.3f} positive_rate={stats['positive_rate']:.3f}"
            )
    for note in notes:
        print(f"note: {note}")
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CLIP linear detector head.")
    parser.add_argument("--features", required=True, type=Path, help="Feature .npz from clip_features.")
    parser.add_argument("--out", type=Path, default=default_settings.detector_path)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularisation strength.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--threshold-metric",
        choices=("balanced_accuracy", "f1", "accuracy"),
        default="balanced_accuracy",
    )
    args = parser.parse_args(argv)
    try:
        train_detector(
            features_path=args.features,
            out_path=args.out,
            C=args.C,
            seed=args.seed,
            calibrate=args.calibrate,
            threshold_metric=args.threshold_metric,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
