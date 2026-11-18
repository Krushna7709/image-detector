"""CLIP feature extraction for the linear detector and attribution heads.

Run as a module::

    python -m aiforensics.training.clip_features \\
        --manifest data/index.csv --out artifacts/clip_features.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aiforensics.analysis.clip_embedder import CLIPEmbedder
from aiforensics.config import Settings, settings as default_settings
from aiforensics.training.augmentations import load_rgb
from aiforensics.training.manifest import (
    ImageRecord,
    ManifestError,
    read_manifest,
    resolve_splits,
)


def save_features(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    generators: np.ndarray,
    splits: np.ndarray,
    paths: np.ndarray,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=features.astype(np.float32),
        labels=labels.astype(np.int8),
        generators=generators.astype(str),
        splits=splits.astype(str),
        paths=paths.astype(str),
    )
    return destination


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "features": data["features"],
            "labels": data["labels"],
            "generators": data["generators"],
            "splits": data["splits"],
            "paths": data["paths"],
        }


def extract_features(
    records: list[ImageRecord],
    settings: Settings | None = None,
    batch_size: int = 16,
    splits: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    selected = [
        record for record in records if splits is None or record.split in splits
    ]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError("No records selected for feature extraction.")

    embedder = CLIPEmbedder(settings or default_settings)
    if not embedder.ensure_loaded():
        raise RuntimeError(
            "CLIP backbone unavailable: "
            f"{embedder.load_error or 'unknown error'}. "
            "Install transformers/torch and make the model available offline or online."
        )

    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    generator_rows: list[str] = []
    split_rows: list[str] = []
    path_rows: list[str] = []
    skipped = 0

    for start in range(0, len(selected), batch_size):
        chunk = selected[start:start + batch_size]
        images = []
        kept: list[ImageRecord] = []
        for record in chunk:
            try:
                images.append(load_rgb(record.path))
                kept.append(record)
            except Exception as exc:
                skipped += 1
                print(f"skip {record.path}: {exc}", file=sys.stderr)
        if not images:
            continue
        vectors = embedder.embed_batch(images, batch_size=batch_size)
        if vectors is None:
            raise RuntimeError(f"CLIP embedding failed: {embedder.load_error}")
        feature_rows.append(vectors)
        for record in kept:
            label_rows.append(1 if record.is_ai else 0)
            generator_rows.append(record.generator)
            split_rows.append(record.split)
            path_rows.append(str(record.path))
        done = min(start + batch_size, len(selected))
        print(f"embedded {done}/{len(selected)}", file=sys.stderr)

    if not feature_rows:
        raise RuntimeError("Every image failed to load; nothing to save.")
    return {
        "features": np.concatenate(feature_rows, axis=0).astype(np.float32),
        "labels": np.asarray(label_rows, dtype=np.int8),
        "generators": np.asarray(generator_rows),
        "splits": np.asarray(split_rows),
        "paths": np.asarray(path_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings for a manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=None, choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None, help="Debug: only the first N images.")
    args = parser.parse_args(argv)

    try:
        records = read_manifest(args.manifest)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not any(record.split for record in records):
        records = resolve_splits(records)

    try:
        data = extract_features(
            records,
            batch_size=max(1, args.batch_size),
            splits=tuple(args.splits) if args.splits else None,
            limit=args.limit,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    destination = save_features(
        args.out,
        data["features"],
        data["labels"],
        data["generators"],
        data["splits"],
        data["paths"],
    )
    unique, counts = np.unique(data["splits"], return_counts=True)
    split_summary = ", ".join(f"{name}={count}" for name, count in zip(unique, counts))
    print(f"Saved {len(data['labels'])} embeddings of dimension {data['features'].shape[1]} to {destination}")
    print(f"Split counts: {split_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
