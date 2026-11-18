"""Train the open-set generator attribution head.

The model is a multi-class logistic regression over CLIP embeddings. A softmax
confidence threshold separates known generators from the ``unknown generator``
class at inference time.

    python -m aiforensics.training.train_attribution \\
        --features artifacts/clip.npz --out models/attribution.joblib
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aiforensics.config import settings as default_settings
from aiforensics.training.clip_features import load_features


def _select(data: dict[str, np.ndarray], split: str):
    mask = data["splits"] == split
    return data["features"][mask], data["labels"][mask], data["generators"][mask]


def _max_probabilities(estimator, features: np.ndarray) -> np.ndarray:
    return np.asarray(estimator.predict_proba(features)).max(axis=1)


def _confidence_floor(known_max: np.ndarray, unknown_max: np.ndarray | None) -> float:
    if known_max.size == 0:
        return 0.4
    if unknown_max is not None and unknown_max.size:
        candidates = np.unique(np.concatenate([known_max, unknown_max, [0.2, 0.4, 0.6, 0.8]]))
        best_score = -1.0
        best_value = 0.4
        for threshold in candidates:
            accepted = float(np.mean(known_max >= threshold))
            rejected = float(np.mean(unknown_max < threshold))
            score = (accepted + rejected) / 2.0
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and abs(threshold - 0.4) < abs(best_value - 0.4)
            ):
                best_score = score
                best_value = float(threshold)
        return float(best_value)
    # Only known-class confidence is observable: accept 95% of it.
    return float(np.clip(np.quantile(known_max, 0.05), 0.3, 0.9))


def _traceability_report(
    probabilities: np.ndarray,
    generators: np.ndarray,
    classes: list[str],
    trained_classes: set[str],
    threshold: float,
) -> dict[str, object]:
    if probabilities.size == 0:
        return {"n": 0}
    predicted_names = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    top_confidence = probabilities.max(axis=1)
    known = np.isin(generators, list(trained_classes))
    accepted = top_confidence >= threshold
    report: dict[str, object] = {
        "n": int(len(generators)),
        "top1_accuracy": round(float(np.mean(predicted_names == generators)), 4),
        "accepted_rate": round(float(np.mean(accepted)), 4),
        "mean_top_confidence": round(float(top_confidence.mean()), 4),
        "threshold": round(float(threshold), 4),
        "known_generators": {},
        "unknown_generators": {},
    }
    for generator in sorted(set(generators.tolist())):
        mask = generators == generator
        stats = {
            "n": int(mask.sum()),
            "top1_accuracy": round(float(np.mean(predicted_names[mask] == generators[mask])), 4),
            "mean_confidences": round(float(top_confidence[mask].mean()), 4),
            "accepted_rate": round(float(np.mean(accepted[mask])), 4),
        }
        bucket = report["known_generators"] if generator in trained_classes else report["unknown_generators"]
        bucket[generator] = stats
    report["known_accepted_rate"] = (
        round(float(np.mean(accepted[known])), 4) if known.any() else None
    )
    report["unknown_rejection_rate"] = (
        round(float(np.mean(~accepted[~known])), 4) if (~known).any() else None
    )
    return report


def train_attribution(
    features_path: Path,
    out_path: Path,
    C: float = 1.0,
    seed: int = 13,
    threshold: float | None = None,
) -> dict[str, object]:
    data = load_features(features_path)
    x_train, y_train, generators_train = _select(data, "train")
    x_val, y_val, generators_val = _select(data, "val")
    x_test, y_test, generators_test = _select(data, "test")
    ai_train = y_train == 1
    if ai_train.sum() == 0:
        raise RuntimeError("No AI images in the training split; attribution needs generator labels.")
    x_train = x_train[ai_train]
    generators_train = generators_train[ai_train]
    ai_val = y_val == 1
    x_val = x_val[ai_val]
    generators_val = generators_val[ai_val]
    ai_test = y_test == 1
    x_test = x_test[ai_test]
    generators_test = generators_test[ai_test]

    classes = sorted(set(generators_train.tolist()))
    if len(classes) < 2:
        raise RuntimeError("Attribution needs at least two AI generators in the training split.")
    class_index = {name: index for index, name in enumerate(classes)}
    y_encoded = np.asarray([class_index[name] for name in generators_train], dtype=np.int64)
    counts = np.bincount(y_encoded, minlength=len(classes))
    if counts.min() < 2:
        weak = [classes[index] for index, count in enumerate(counts) if count < 2]
        raise RuntimeError(f"Each generator needs at least two training images; add more for: {', '.join(weak)}")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_encoded)

    notes: list[str] = []
    if threshold is not None:
        open_set_threshold = float(np.clip(threshold, 0.01, 0.99))
        threshold_source = "cli"
    else:
        known_val = np.isin(generators_val, classes)
        unknown_val = ~known_val
        if known_val.any():
            known_max = _max_probabilities(pipeline, x_val[known_val])
            unknown_max = _max_probabilities(pipeline, x_val[unknown_val]) if unknown_val.any() else None
            open_set_threshold = _confidence_floor(known_max, unknown_max)
            threshold_source = "val"
        else:
            folds = int(max(2, min(5, counts.min())))
            cross_val = cross_val_predict(
                pipeline,
                x_train,
                y_encoded,
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
                method="predict_proba",
            )
            open_set_threshold = _confidence_floor(cross_val.max(axis=1), None)
            threshold_source = f"{folds}-fold-cross-validation"
            notes.append(
                "Validation split contains only unseen generators; the open-set threshold "
                "was derived from out-of-fold training confidences."
            )
        if open_set_threshold > 0.8:
            notes.append(
                "Open-set threshold is high because training scores are overconfident; "
                "collect known-generator validation images to tune it properly."
            )

    reports = {}
    for split_name, features, generators in (
        ("val", x_val, generators_val),
        ("test", x_test, generators_test),
    ):
        if features.size == 0:
            continue
        probabilities = np.asarray(pipeline.predict_proba(features))
        reports[split_name] = _traceability_report(
            probabilities,
            generators,
            classes,
            set(classes),
            open_set_threshold,
        )

    artifact = {
        "pipeline": pipeline,
        "classes": classes,
        "threshold": float(open_set_threshold),
        "trained_on": classes,
        "metrics": reports,
        "threshold_source": threshold_source,
        "feature": "clip",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "C": C,
        "notes": notes,
    }
    destination = Path(out_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)

    print(
        f"Saved attribution head to {destination} with {len(classes)} generators "
        f"(threshold={open_set_threshold:.3f}, source={threshold_source})"
    )
    print(f"Generators: {', '.join(classes)}")
    for split_name, report in reports.items():
        print(
            f"{split_name}: n={report['n']} top1={report['top1_accuracy']:.3f} "
            f"accepted={report['accepted_rate']:.3f} unknown_rejection={report['unknown_rejection_rate']}"
        )
        for generator, stats in report["unknown_generators"].items():
            print(f"  unseen {generator:20s} accepted_rate={stats['accepted_rate']:.3f}")
    for note in notes:
        print(f"note: {note}")
    print(json.dumps(reports, indent=2))
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the open-set attribution head.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=default_settings.attribution_path)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the open-set confidence threshold instead of deriving it.",
    )
    args = parser.parse_args(argv)
    try:
        train_attribution(
            features_path=args.features,
            out_path=args.out,
            C=args.C,
            seed=args.seed,
            threshold=args.threshold,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
