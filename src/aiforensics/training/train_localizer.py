"""Train the U-Net edit localizer on image/mask pairs.

Produces ``models/localizer.pt`` for ``UNetLocalizer``. Manifest rows need a
``mask`` column pointing at a binary manipulation mask::

    python -m aiforensics.training.train_localizer \\
        --manifest data/edits.csv --epochs 15 --out models/localizer.pt
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aiforensics.training.datasets import LocalizationDataset, seed_worker
from aiforensics.training.manifest import (
    ImageRecord,
    ManifestError,
    read_manifest,
    resolve_splits,
    split_records,
)
from aiforensics.training.metrics import aggregate_mask_metrics, mask_metrics


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _masked(records: list[ImageRecord]) -> list[ImageRecord]:
    return [record for record in records if record.mask is not None]


def _ensure_validation(
    train: list[ImageRecord], val: list[ImageRecord], test: list[ImageRecord], seed: int
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    if val:
        return train, val, test
    if test:
        print("note: no validation split; selecting the checkpoint on the test split (optimistic).")
        return train, test, []
    if len(train) < 4:
        raise RuntimeError("Need a validation split or at least four masked training images.")
    shuffled = list(train)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(0.1 * len(shuffled))))
    return shuffled[n_val:], shuffled[:n_val], []


def _segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probabilities * target).sum(dims)
    union = probabilities.sum(dims) + target.sum(dims)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
    return bce + dice


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    optimizer=None,
    scaler=None,
    amp: bool = False,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    per_image: list[dict[str, float]] = []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = _segmentation_loss(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
        batch = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch
        total_items += batch
        predicted = (torch.sigmoid(logits.detach()) > threshold).float().cpu().numpy()
        target = (masks.detach() > 0.5).float().cpu().numpy()
        for index in range(predicted.shape[0]):
            metrics = mask_metrics(predicted[index, 0], target[index, 0])
            if metrics is not None:
                per_image.append(metrics)
    loss_value = total_loss / total_items if total_items else 0.0
    summary = aggregate_mask_metrics(per_image)
    summary["loss"] = round(loss_value, 4)
    return loss_value, summary


def train_localizer(
    manifest_path: Path,
    out_path: Path,
    width: int = 32,
    size: int = 512,
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    workers: int = 2,
    device_name: str = "auto",
    amp: bool = True,
    seed: int = 13,
    limit: int | None = None,
    mask_threshold: float = 0.5,
    metrics_out: Path | None = None,
) -> dict[str, object]:
    from aiforensics.models.unet import build_unet

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _select_device(device_name)

    records = _masked(read_manifest(manifest_path))
    if not records:
        raise RuntimeError("No manifest rows have a mask; the localizer needs edit masks.")
    if not any(record.split for record in records):
        print("note: manifest has no splits; creating a random 70/15/15 split.")
        records = resolve_splits(records, seed=seed)
    train_records = _masked(split_records(records, "train"))
    val_records = _masked(split_records(records, "val"))
    test_records = _masked(split_records(records, "test"))
    if len(train_records) < 2:
        raise RuntimeError("At least two masked training images are required.")
    train_records, val_records, test_records = _ensure_validation(
        train_records, val_records, test_records, seed
    )
    if limit is not None:
        train_records = train_records[:limit]
        val_records = val_records[:limit]
        test_records = test_records[:limit]

    amp_enabled = bool(amp and device.type == "cuda")
    if amp and not amp_enabled:
        print("note: AMP is only enabled on CUDA; training on CPU in float32.")

    train_loader = DataLoader(
        LocalizationDataset(train_records, size=size, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_records) > batch_size,
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        LocalizationDataset(val_records, size=size, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
    )
    test_loader = (
        DataLoader(
            LocalizationDataset(test_records, size=size, train=False),
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

    model = build_unet(width=width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(
        f"Training U-Net (width={width}) on {len(train_records)} masked images "
        f"for {epochs} epochs on {device}."
    )
    best_score = -float("inf")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model, train_loader, device, mask_threshold, optimizer, scaler, amp_enabled
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, device, mask_threshold
        )
        score = float(val_metrics.get("iou", 0.0))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_iou": train_metrics.get("iou"),
                "val_loss": round(val_loss, 4),
                "val_iou": val_metrics.get("iou"),
            }
        )
        marker = ""
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            marker = " *"
        print(
            f"epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} "
            f"train_iou={train_metrics.get('iou')} val_loss={val_loss:.4f} "
            f"val_iou={val_metrics.get('iou')}{marker}"
        )

    model.load_state_dict(best_state)
    val_loss, val_metrics = _run_epoch(model, val_loader, device, mask_threshold)
    test_metrics = None
    if test_loader is not None:
        test_loss, test_summary = _run_epoch(
            model, test_loader, device, mask_threshold
        )
        test_metrics = dict(test_summary)
        test_metrics["loss"] = round(test_loss, 4)

    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "width": width,
        "input_size": size,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "history": history,
        "trained_on": "synthetic edit masks",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "amp": amp_enabled,
    }
    destination = Path(out_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)

    print(f"Saved localizer to {destination}")
    print(
        f"val: iou={val_metrics.get('iou')} dice={val_metrics.get('dice')} "
        f"precision={val_metrics.get('precision')} recall={val_metrics.get('recall')}"
    )
    if test_metrics is not None:
        print(
            f"test: iou={test_metrics.get('iou')} dice={test_metrics.get('dice')} "
            f"images={test_metrics.get('images')}"
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
    parser = argparse.ArgumentParser(description="Train the U-Net edit localizer.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("models") / "localizer.pt")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap images per split.")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--metrics-out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        train_localizer(
            manifest_path=args.manifest,
            out_path=args.out,
            width=args.width,
            size=args.size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            workers=args.workers,
            device_name=args.device,
            amp=args.amp,
            seed=args.seed,
            limit=args.limit,
            mask_threshold=args.mask_threshold,
            metrics_out=args.metrics_out,
        )
    except (RuntimeError, ManifestError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
