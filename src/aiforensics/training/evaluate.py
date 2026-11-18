"""Evaluate the assembled engine on a labelled manifest.

Unlike the head-only training metrics, this measures the product behaviour:
fusion, provenance, open-set attribution and robustness distortions.

    python -m aiforensics.training.evaluate \\
        --manifest data/index.csv --splits test \\
        --output artifacts/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from aiforensics.config import settings as default_settings
from aiforensics.core.engine import ForensicsEngine
from aiforensics.training.manifest import (
    ImageRecord,
    ManifestError,
    read_manifest,
    resolve_splits,
)
from aiforensics.training.metrics import (
    aggregate_mask_metrics,
    classification_metrics,
    confusion_matrix_text,
    mask_metrics,
)

DEFAULT_ROBUSTNESS = ("jpeg50", "jpeg30", "resize50", "blur2")


def _row_from_result(record: ImageRecord, distortion: str, result) -> dict[str, object]:
    items = result.attribution
    top = items[0] if items else None
    hint = result.metadata.generator_hint if result.metadata else None
    return {
        "path": str(record.path),
        "split": record.split,
        "label": record.label,
        "generator": record.generator,
        "distortion": distortion,
        "probability": float(result.verdict.ai_probability),
        "predicted_label": result.verdict.label,
        "attribution_top": top.generator if top else "",
        "attribution_source": top.source if top else "none",
        "attribution_known": bool(top.known) if top else False,
        "attribution_confidence": float(top.confidence) if top else 0.0,
        "attribution_correct": bool(top and top.known and top.generator == record.generator),
        "metadata_hint": hint,
        "metadata_hint_correct": bool(hint and hint == record.generator),
        "edited_probability": float(result.edit.edited_probability),
        "edit_regions": len(result.edit.regions),
        "engine": result.engine,
    }


def _summary(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    if not rows:
        return {"n": 0}
    labels = np.asarray([1 if row["label"] == "ai" else 0 for row in rows], dtype=np.int64)
    probabilities = np.asarray([row["probability"] for row in rows], dtype=np.float32)
    metrics = classification_metrics(labels, probabilities, threshold)
    low, high = default_settings.uncertain_low, default_settings.uncertain_high
    metrics["uncertain_rate"] = round(
        float(np.mean((probabilities >= low) & (probabilities < high))), 4
    )
    metrics["mean_probability"] = round(float(probabilities.mean()), 4)
    return metrics


def _per_generator(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    report: dict[str, object] = {}
    for generator in sorted({str(row["generator"]) for row in rows}):
        subset = [row for row in rows if row["generator"] == generator]
        probabilities = np.asarray([row["probability"] for row in subset], dtype=np.float32)
        is_ai = subset[0]["label"] == "ai"
        stats: dict[str, object] = {
            "label": "ai" if is_ai else "real",
            "n": len(subset),
            "mean_probability": round(float(probabilities.mean()), 4),
            "uncertain_rate": round(
                float(
                    np.mean(
                        (probabilities >= default_settings.uncertain_low)
                        & (probabilities < default_settings.uncertain_high)
                    )
                ),
                4,
            ),
            "mean_edited_probability": round(
                float(np.mean([row["edited_probability"] for row in subset])), 4
            ),
            "mean_regions": round(float(np.mean([row["edit_regions"] for row in subset])), 4),
        }
        if is_ai:
            stats["detection_rate"] = round(float(np.mean(probabilities >= threshold)), 4)
            stats["attribution_top1"] = round(
                float(np.mean([row["attribution_correct"] for row in subset])), 4
            )
            stats["metadata_hint_rate"] = round(
                float(np.mean([bool(row["metadata_hint"]) for row in subset])), 4
            )
        else:
            stats["false_positive_rate"] = round(float(np.mean(probabilities >= threshold)), 4)
            stats["unknown_attribution_rate"] = round(
                float(np.mean([row["attribution_top"] == "unknown generator" for row in subset])),
                4,
            )
        report[generator] = stats
    return report


def _attribution_summary(rows: list[dict[str, object]], engine: ForensicsEngine) -> dict[str, object]:
    ai_rows = [row for row in rows if row["label"] == "ai"]
    if not ai_rows:
        return {"n": 0}
    known_classes = list(engine.attribution.classes) if engine.attribution is not None else []
    hint_rows = [row for row in ai_rows if row["metadata_hint"]]
    model_rows = [row for row in ai_rows if row["attribution_source"] == "model"]
    unseen_rows = (
        [row for row in ai_rows if row["generator"] not in known_classes] if known_classes else []
    )
    summary: dict[str, object] = {
        "n": len(ai_rows),
        "trained_classes": known_classes,
        "top1_exact": round(float(np.mean([row["attribution_correct"] for row in ai_rows])), 4),
        "top1_known_rate": round(float(np.mean([row["attribution_known"] for row in ai_rows])), 4),
        "unknown_reported_rate": round(
            float(np.mean([row["attribution_top"] == "unknown generator" for row in ai_rows])), 4
        ),
        "metadata_hint_rate": round(float(len(hint_rows) / len(ai_rows)), 4),
        "metadata_hint_correct": (
            round(float(np.mean([row["metadata_hint_correct"] for row in hint_rows])), 4)
            if hint_rows
            else None
        ),
        "model_predictions": len(model_rows),
        "model_top1_exact": (
            round(float(np.mean([row["attribution_correct"] for row in model_rows])), 4)
            if model_rows
            else None
        ),
    }
    if unseen_rows:
        summary["unseen_generator_images"] = len(unseen_rows)
        summary["unseen_unknown_reported"] = round(
            float(np.mean([row["attribution_top"] == "unknown generator" for row in unseen_rows])),
            4,
        )
    return summary


def _localization_summary(
    records: list[ImageRecord], engine: ForensicsEngine
) -> dict[str, object]:
    if engine.unet is None:
        return {
            "method": "none",
            "images": 0,
            "note": "No trained U-Net localizer is loaded; heuristic regions were not scored against masks.",
        }
    masked = [record for record in records if record.mask is not None]
    if not masked:
        return {"method": "unet", "images": 0, "note": "No manifest rows include edit masks."}
    values: list[dict[str, float]] = []
    failures: list[dict[str, str]] = []
    for record in masked:
        try:
            with Image.open(record.path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            with Image.open(record.mask) as handle:
                mask_image = handle.convert("L")
            if mask_image.size != image.size:
                mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
            heat = engine.unet.predict(np.asarray(image))
            metrics = mask_metrics(heat > 0.5, np.asarray(mask_image) > 127)
            if metrics is not None:
                values.append(metrics)
        except Exception as exc:
            failures.append({"path": str(record.path), "error": str(exc)})
    summary = aggregate_mask_metrics(values)
    summary["method"] = "unet"
    if failures:
        summary["failures"] = failures[:10]
    return summary


def evaluate_engine(
    manifest_path: Path,
    splits: tuple[str, ...] | None = None,
    limit: int | None = None,
    robustness: tuple[str, ...] = DEFAULT_ROBUSTNESS,
    localization: bool = True,
) -> dict[str, object]:
    records = read_manifest(manifest_path)
    if not any(record.split for record in records):
        print("note: manifest has no splits; creating a random split for evaluation.")
        records = resolve_splits(records)
    available = sorted({record.split for record in records if record.split})
    if splits:
        selected_splits = tuple(split for split in splits if split in available)
        if not selected_splits:
            raise RuntimeError(f"None of the requested splits exist; available: {', '.join(available)}")
    elif "test" in available:
        selected_splits = ("test",)
    elif "val" in available:
        selected_splits = ("val",)
    else:
        selected_splits = tuple(available)
    selected = [record for record in records if record.split in selected_splits]
    if not selected:
        raise RuntimeError("No records selected for evaluation.")
    if limit is not None:
        selected = selected[:limit]

    engine = ForensicsEngine()
    health = engine.health().model_dump()
    print(f"Engine: detector={health['detector']} attribution={health['attribution']} localizer={health['localizer']}")
    print(f"Selected splits: {', '.join(selected_splits)} ({len(selected)} images)")

    primary_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for index, record in enumerate(selected, start=1):
        try:
            result = engine.analyze(record.path.read_bytes(), record.path.name, distortion="none")
            primary_rows.append(_row_from_result(record, "none", result))
        except Exception as exc:
            errors.append({"path": str(record.path), "error": f"{type(exc).__name__}: {exc}"})
        if index % 50 == 0:
            print(f"analysed {index}/{len(selected)}")

    if not primary_rows:
        raise RuntimeError("Every image failed to analyse; nothing to evaluate.")
    threshold = default_settings.decision_threshold

    robustness_rows: dict[str, list[dict[str, object]]] = {}
    for distortion in robustness:
        if distortion in {"none", ""}:
            continue
        rows: list[dict[str, object]] = []
        for record in selected:
            try:
                result = engine.analyze(record.path.read_bytes(), record.path.name, distortion=distortion)
                rows.append(_row_from_result(record, distortion, result))
            except Exception as exc:
                errors.append(
                    {
                        "path": str(record.path),
                        "distortion": distortion,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        robustness_rows[distortion] = rows
        print(f"robustness {distortion}: {len(rows)} images")

    by_split = {
        split: _summary([row for row in primary_rows if row["split"] == split], threshold)
        for split in selected_splits
    }
    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(Path(manifest_path).resolve()),
        "engine": health,
        "splits": list(selected_splits),
        "decision_threshold": threshold,
        "overall": _summary(primary_rows, threshold),
        "by_split": by_split,
        "per_generator": _per_generator(primary_rows, threshold),
        "attribution": _attribution_summary(primary_rows, engine),
        "edit_heuristic": {
            "real_mean_edited_probability": round(
                float(np.mean([row["edited_probability"] for row in primary_rows if row["label"] == "real"])), 4
            )
            if any(row["label"] == "real" for row in primary_rows)
            else None,
            "ai_mean_edited_probability": round(
                float(np.mean([row["edited_probability"] for row in primary_rows if row["label"] == "ai"])), 4
            )
            if any(row["label"] == "ai" for row in primary_rows)
            else None,
        },
        "robustness": {
            distortion: _summary(rows, threshold) for distortion, rows in robustness_rows.items() if rows
        },
        "errors": errors,
    }
    if localization:
        report["localization"] = _localization_summary(selected, engine)

    overall = report["overall"]
    print(
        f"overall: n={overall['n']} accuracy={overall['accuracy']:.3f} "
        f"f1={overall['f1']:.3f} auc={overall['roc_auc']} ece={overall['ece']:.3f} "
        f"uncertain={overall['uncertain_rate']:.3f}"
    )
    print(confusion_matrix_text(overall["confusion"]))
    for split, metrics in by_split.items():
        if metrics.get("n"):
            print(
                f"  {split:5s} n={metrics['n']:4d} accuracy={metrics['accuracy']:.3f} "
                f"f1={metrics['f1']:.3f} auc={metrics['roc_auc']} ece={metrics['ece']:.3f}"
            )
    print("per generator:")
    for generator, stats in report["per_generator"].items():
        extra = (
            f"detection={stats['detection_rate']:.3f}"
            if "detection_rate" in stats
            else f"false_positive={stats['false_positive_rate']:.3f}"
        )
        print(
            f"  {generator:20s} n={stats['n']:4d} mean_p={stats['mean_probability']:.3f} "
            f"{extra} attribution_top1={stats.get('attribution_top1')}"
        )
    if report["robustness"]:
        print("robustness:")
        for distortion, metrics in report["robustness"].items():
            print(
                f"  {distortion:10s} n={metrics['n']:4d} accuracy={metrics['accuracy']:.3f} "
                f"f1={metrics['f1']:.3f} auc={metrics['roc_auc']} ece={metrics['ece']:.3f}"
            )
    if errors:
        print(f"note: {len(errors)} analyses failed; see report['errors'].")
    counts = Counter(str(row["engine"]) for row in primary_rows)
    print(f"engines used: {dict(counts)}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the full ForensIQ engine.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--robustness",
        nargs="*",
        default=list(DEFAULT_ROBUSTNESS),
        choices=(*DEFAULT_ROBUSTNESS, "none"),
        help="Distortions to sweep (pass nothing to disable).",
    )
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-localization", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    robustness = () if args.skip_robustness else tuple(args.robustness or ())
    try:
        report = evaluate_engine(
            manifest_path=args.manifest,
            splits=tuple(args.splits) if args.splits else None,
            limit=args.limit,
            robustness=robustness,
            localization=not args.skip_localization,
        )
    except (RuntimeError, ManifestError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
