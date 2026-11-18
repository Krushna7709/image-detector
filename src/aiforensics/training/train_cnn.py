"""Fine-tune the CNN detector on a manifest.

Produces ``models/cnn_detector.pt`` in the format consumed by
``aiforensics.analysis.cnn_detector.CNNDetector``::

    python -m aiforensics.training.train_cnn \\
        --manifest data/index.csv --arch smallcnn --epochs 10 --out models/cnn_detector.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aiforensics.training.augmentations import IMAGENET_MEAN, IMAGENET_STD
from aiforensics.training.datasets import ClassificationDataset, seed_worker
from aiforensics.training.manifest import (
    ImageRecord,
    ManifestError,
    read_manifest,
    resolve_splits,
    split_records,
)
from aiforensics.training.metrics import (
    classification_metrics,
    confusion_matrix_text,
    expected_calibration_error,
)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _ensure_validation(
    train: list[ImageRecord], val: list[ImageRecord], test: list[ImageRecord], seed: int
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    if val:
        return train, val, test
    if test:
        print("note: no validation split; selecting the checkpoint on the test split (optimistic).")
        return train, test, []
    if len(train) < 4:
        raise RuntimeError("Need a validation split or at least four training images.")
    shuffled = list(train)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(0.1 * len(shuffled))))
    return shuffled[n_val:], shuffled[:n_val], []


def _generator_report(
    labels: np.ndarray, probabilities: np.ndarray, generators: np.ndarray, threshold: float
) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for generator in sorted(set(generators.tolist())):
        mask = generators == generator
        report[generator] = {
            "n": int(mask.sum()),
            "ai_label_rate": round(float(labels[mask].mean()), 4),
            "mean_probability": round(float(probabilities[mask].mean()), 4),
            "positive_rate": round(float((probabilities[mask] >= threshold).mean()), 4),
        }
    return report


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    optimizer=None,
    scaler=None,
    amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        batch = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch
        total_items += batch
        all_probabilities.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    if total_items == 0:
        return 0.0, np.zeros(0), np.zeros(0)
    return (
        total_loss / total_items,
        np.concatenate(all_probabilities).ravel(),
        np.concatenate(all_labels).ravel(),
    )


def train_cnn(
    manifest_path: Path,
    out_path: Path,
    arch: str = "smallcnn",
    width: int = 32,
    size: int = 256,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    backbone_learning_rate: float | None = None,
    weight_decay: float = 1e-4,
    workers: int = 2,
    device_name: str = "auto",
    amp: bool = True,
    seed: int = 13,
    limit: int | None = None,
    pretrained: bool = False,
    metrics_out: Path | None = None,
) -> dict[str, object]:
    from aiforensics.models.cnn import build_detector

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _select_device(device_name)

    records = read_manifest(manifest_path)
    if not any(record.split for record in records):
        print("note: manifest has no splits; creating a random 70/15/15 split.")
        records = resolve_splits(records, seed=seed)
    train_records = split_records(records, "train")
    val_records = split_records(records, "val")
    test_records = split_records(records, "test")
    if not train_records:
        raise RuntimeError("Manifest contains no 'train' records.")
    train_records, val_records, test_records = _ensure_validation(
        train_records, val_records, test_records, seed
    )
    if limit is not None:
        train_records = train_records[:limit]
        val_records = val_records[:limit]
        test_records = test_records[:limit]

    train_ai = sum(record.is_ai for record in train_records)
    train_real = len(train_records) - train_ai
    if not train_ai or not train_real:
        raise RuntimeError("Training data needs both real and AI images.")

    amp_enabled = bool(amp and device.type == "cuda")
    if amp and not amp_enabled:
        print("note: AMP is only enabled on CUDA; training on CPU in float32.")

    train_loader = DataLoader(
        ClassificationDataset(train_records, size=size, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_records) > batch_size,
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        ClassificationDataset(val_records, size=size, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
    )
    test_loader = (
        DataLoader(
            ClassificationDataset(test_records, size=size, train=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            persistent_workers=workers > 0,
        )
        if test_records
        else None
    )

    model = build_detector(arch, width=width, pretrained=pretrained).to(device)
    pos_weight = torch.tensor([train_real / train_ai], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if backbone_learning_rate is None and arch != "smallcnn":
        backbone_learning_rate = learning_rate * 0.1
    if backbone_learning_rate is not None:
        backbone_parameters = [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith("fc.") and parameter.requires_grad
        ]
        head_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name.startswith("fc.") and parameter.requires_grad
        ]
        parameter_groups = [
            {"params": head_parameters, "lr": learning_rate},
            {"params": backbone_parameters, "lr": backbone_learning_rate},
        ]
        parameter_groups = [group for group in parameter_groups if group["params"]]
        optimizer = torch.optim.AdamW(
            parameter_groups, lr=learning_rate, weight_decay=weight_decay
        )
        print(
            f"Optimizer: head lr={learning_rate:g}, "
            f"backbone lr={backbone_learning_rate:g}."
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(
        f"Training {arch} on {len(train_records)} images "
        f"(real={train_real}, ai={train_ai}) for {epochs} epochs on {device}."
    )
    best_score = -float("inf")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_probs, train_labels = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_enabled
        )
        val_loss, val_probs, val_labels = _run_epoch(model, val_loader, criterion, device)
        val_metrics = classification_metrics(val_labels, val_probs, 0.5)
        train_accuracy = float(np.mean((train_probs >= 0.5) == (train_labels >= 0.5)))
        score = (
            float(val_metrics["roc_auc"])
            if val_metrics.get("roc_auc") is not None
            else -val_loss
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_accuracy, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": val_metrics["accuracy"],
                "val_auc": val_metrics.get("roc_auc"),
            }
        )
        marker = ""
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            marker = " *"
        print(
            f"epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.3f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_auc={val_metrics.get('roc_auc')}{marker}"
        )

    model.load_state_dict(best_state)
    val_loss, val_probs, val_labels = _run_epoch(model, val_loader, criterion, device)
    val_generators = np.asarray([record.generator for record in val_records])
    val_metrics = classification_metrics(val_labels, val_probs, 0.5)
    val_metrics["per_generator"] = _generator_report(
        val_labels, val_probs, val_generators, 0.5
    )
    val_metrics["loss"] = round(val_loss, 4)
    test_metrics = None
    if test_loader is not None:
        test_loss, test_probs, test_labels = _run_epoch(model, test_loader, criterion, device)
        test_generators = np.asarray([record.generator for record in test_records])
        test_metrics = classification_metrics(test_labels, test_probs, 0.5)
        test_metrics["per_generator"] = _generator_report(
            test_labels, test_probs, test_generators, 0.5
        )
        test_metrics["loss"] = round(test_loss, 4)

    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "arch": arch,
        "width": width,
        "input_size": size,
        "mean": IMAGENET_MEAN.tolist(),
        "std": IMAGENET_STD.tolist(),
        "target_layer": None,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "history": history,
        "trained_on": sorted(set(record.generator for record in train_records if record.is_ai)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "amp": amp_enabled,
        "learning_rate": learning_rate,
        "backbone_learning_rate": backbone_learning_rate,
    }
    destination = Path(out_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)

    print(f"Saved CNN checkpoint to {destination}")
    print(
        f"val: acc={val_metrics['accuracy']:.3f} auc={val_metrics.get('roc_auc')} "
        f"ece={expected_calibration_error(val_labels, val_probs):.3f}"
    )
    print(confusion_matrix_text(val_metrics["confusion"]))
    if test_metrics is not None:
        print(
            f"test: acc={test_metrics['accuracy']:.3f} auc={test_metrics.get('roc_auc')} "
            f"ece={test_metrics['ece']:.3f}"
        )
        print(confusion_matrix_text(test_metrics["confusion"]))
        for generator, stats in test_metrics["per_generator"].items():
            print(
                f"  {generator:20s} n={stats['n']:5d} "
                f"mean_p={stats['mean_probability']:.3f} positive_rate={stats['positive_rate']:.3f}"
            )
    if metrics_out:
        metrics_path = Path(metrics_out).expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps({"val": val_metrics, "test": test_metrics, "history": history}, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote metrics to {metrics_path}")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the CNN detector.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("models") / "cnn_detector.pt")
    parser.add_argument("--arch", choices=("smallcnn", "convnext_tiny"), default="smallcnn")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Backbone learning rate (defaults to 0.1x the head rate for ConvNeXt).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap images per split.")
    parser.add_argument("--pretrained", action="store_true", help="Start from torchvision weights.")
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        train_cnn(
            manifest_path=args.manifest,
            out_path=args.out,
            arch=args.arch,
            width=args.width,
            size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            backbone_learning_rate=args.backbone_lr,
            weight_decay=args.weight_decay,
            workers=args.workers,
            device_name=args.device,
            amp=args.amp,
            seed=args.seed,
            limit=args.limit,
            pretrained=args.pretrained,
            metrics_out=args.metrics_out,
        )
    except (RuntimeError, ManifestError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
