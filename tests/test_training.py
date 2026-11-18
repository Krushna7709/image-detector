from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aiforensics.training.calibration import fit_calibrator
from aiforensics.training.clip_features import load_features, save_features
from aiforensics.training.make_edits import make_edits
from aiforensics.training.manifest import (
    ImageRecord,
    build_manifest,
    read_manifest,
    resolve_splits,
    summarize,
)
from aiforensics.training.metrics import (
    aggregate_mask_metrics,
    best_threshold,
    classification_metrics,
    expected_calibration_error,
    mask_metrics,
)
from aiforensics.training.train_attribution import train_attribution
from aiforensics.training.train_detector import train_detector


def _write_detector_features(path, seed: int = 3):
    rng = np.random.default_rng(seed)
    rows = []
    for split, count in (("train", 60), ("val", 20), ("test", 20)):
        for index in range(count):
            label = index % 2
            generator = ("genA", "genB")[index % 4 // 2] if label else "real"
            features = rng.normal(size=6) + label * 1.8
            rows.append((features, label, generator, split))
    return save_features(
        path,
        features=np.stack([row[0] for row in rows]),
        labels=np.array([row[1] for row in rows]),
        generators=np.array([row[2] for row in rows]),
        splits=np.array([row[3] for row in rows]),
        paths=np.array([f"image-{index}.png" for index in range(len(rows))]),
    )


def _write_attribution_features(path, seed: int = 4):
    rng = np.random.default_rng(seed)
    rows = []
    for split, count in (("train", 40), ("val", 8), ("test", 8)):
        for index in range(count):
            if split == "train":
                generator = "genA" if index % 2 else "genB"
                centre = 2.0 if generator == "genA" else -2.0
            else:
                generator = "genC"
                centre = 0.0
            rows.append((rng.normal(size=6) + centre, generator, split))
    return save_features(
        path,
        features=np.stack([row[0] for row in rows]),
        labels=np.ones(len(rows), dtype=np.int64),
        generators=np.array([row[1] for row in rows]),
        splits=np.array([row[2] for row in rows]),
        paths=np.array([f"image-{index}.png" for index in range(len(rows))]),
    )


def test_metrics_and_thresholds():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    metrics = classification_metrics(y, p, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion"] == {
        "tp": 2,
        "tn": 2,
        "fp": 0,
        "fn": 0,
        "real": {"real": 2, "ai": 0},
        "ai": {"real": 0, "ai": 2},
    }
    assert best_threshold(y, p) == pytest.approx(0.5)
    assert best_threshold([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert best_threshold([0, 1], [0.5, 0.5], metric="f1") == pytest.approx(0.5)
    assert expected_calibration_error([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert expected_calibration_error([0, 0, 1, 1], [0.2, 0.2, 0.8, 0.8]) == pytest.approx(0.2)


def test_mask_metrics():
    pred = np.zeros((8, 8), dtype=np.uint8)
    pred[2:6, 2:6] = 1
    perfect = mask_metrics(pred, pred)
    assert perfect == {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0}

    empty = mask_metrics(np.zeros_like(pred), pred)
    assert empty == {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    aggregate = aggregate_mask_metrics([perfect, perfect])
    assert aggregate["iou"] == pytest.approx(1.0)
    assert aggregate["images"] == 2


def test_platt_calibration_is_monotonic():
    rng = np.random.default_rng(0)
    probabilities = np.linspace(0.05, 0.95, 200)
    y = (rng.random(probabilities.size) < probabilities).astype(int)
    calibrator = fit_calibrator(probabilities, y)
    assert calibrator is not None
    adjusted = calibrator.predict(probabilities)
    assert np.all(np.diff(adjusted) >= 0)
    assert adjusted.min() >= 0.0 and adjusted.max() <= 1.0


def test_train_detector_and_attribution(tmp_path):
    features_path = _write_detector_features(tmp_path / "features.npz")
    data = load_features(features_path)
    assert data["features"].shape == (100, 6)
    assert set(data["splits"]) == {"train", "val", "test"}

    detector_path = tmp_path / "detector.joblib"
    artifact = train_detector(features_path, detector_path, seed=1)
    assert detector_path.exists()
    assert artifact["calibrated"] is True
    assert artifact["trained_on"] == ["genA", "genB"]
    assert 0.01 <= artifact["threshold"] <= 0.99
    assert artifact["metrics"]["test"]["accuracy"] > 0.8

    attribution_path = tmp_path / "attribution.joblib"
    attribution = train_attribution(features_path, attribution_path, seed=1)
    assert attribution_path.exists()
    assert attribution["classes"] == ["genA", "genB"]
    assert attribution["threshold_source"] in {"val", "2-fold-cross-validation"}
    assert attribution["metrics"]["test"]["n"] == 10


def test_open_set_threshold_from_unseen_generators(tmp_path):
    features_path = _write_attribution_features(tmp_path / "features.npz")
    artifact = train_attribution(features_path, tmp_path / "attribution.joblib", seed=2)
    assert artifact["threshold_source"].endswith("cross-validation")
    report = artifact["metrics"]["test"]
    assert report["unknown_generators"]["genC"]["n"] == 8


def test_manifest_build_is_generator_disjoint(tmp_path, image_factory):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    for index in range(4):
        (real_dir / f"real-{index}.png").write_bytes(image_factory(seed=index, width=96, height=80))

    ai_dirs = {}
    for name in ("alpha", "beta", "gamma"):
        directory = tmp_path / name
        directory.mkdir()
        for index in range(3):
            (directory / f"{name}-{index}.png").write_bytes(
                image_factory(seed=10 + index, width=96, height=80)
            )
        ai_dirs[name] = directory

    manifest_path = tmp_path / "manifest.csv"
    records = build_manifest([real_dir], ai_dirs, manifest_path, seed=5)
    assert manifest_path.exists()
    assert len(records) == 13
    assert len({record.split for record in records if record.is_ai}) == 3

    by_split = {}
    for record in records:
        if record.is_ai:
            by_split.setdefault(record.split, set()).add(record.generator)
    assert set().union(*by_split.values()) == {"alpha", "beta", "gamma"}
    all_ai = [generator for names in by_split.values() for generator in names]
    assert len(all_ai) == len(set(all_ai))

    reloaded = read_manifest(manifest_path)
    assert {record.path for record in reloaded} == {record.path for record in records}
    table = summarize(reloaded)
    assert table["train"]["total"] + table["val"]["total"] + table["test"]["total"] == 13


def test_resolve_splits_stratifies_small_generator_sets(tmp_path):
    records = []
    for index in range(20):
        records.append(
            ImageRecord(path=tmp_path / f"real-{index}.png", label="real", generator="real")
        )
        records.append(
            ImageRecord(
                path=tmp_path / f"ai-{index}.png",
                label="ai",
                generator="genA" if index % 2 else "genB",
            )
        )
    resolved = resolve_splits(records, seed=3)
    assert all(record.split for record in resolved)
    train_labels = {record.label for record in resolved if record.split == "train"}
    assert train_labels == {"real", "ai"}
    assert {record.split for record in resolved} == {"train", "val", "test"}


def test_make_edits_writes_masks(tmp_path, image_factory):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "photo.png").write_bytes(image_factory(seed=9))
    ai_dir = tmp_path / "gen"
    ai_dir.mkdir()
    (ai_dir / "render.png").write_bytes(image_factory(seed=11))

    source_manifest = tmp_path / "source.csv"
    build_manifest([real_dir], {"gen": ai_dir}, source_manifest, seed=1, split=False)

    edited = make_edits(
        source_manifest,
        tmp_path / "edits",
        tmp_path / "masks",
        tmp_path / "edits.csv",
        per_image=2,
        seed=3,
        jpeg_probability=0.0,
    )
    assert len(edited) == 4
    for record in edited:
        assert record.mask is not None
        assert record.path.exists()
        assert record.mask.exists()
        mask = np.asarray(Image.open(record.mask))
        assert int(mask.max()) == 255
    assert len(read_manifest(tmp_path / "edits.csv")) == 4
