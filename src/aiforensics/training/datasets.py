"""Torch datasets backed by dataset manifests."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aiforensics.training.augmentations import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    eval_image,
    load_rgb,
    resize,
    to_normalized_array,
    train_classification_image,
    train_localization_pair,
)
from aiforensics.training.manifest import ImageRecord


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - required signature
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ClassificationDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        size: int = 256,
        train: bool = False,
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray = IMAGENET_STD,
    ):
        self.records = list(records)
        self.size = size
        self.train = train
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        try:
            image = load_rgb(record.path)
        except Exception as exc:  # pragma: no cover - filesystem dependent
            raise RuntimeError(f"Could not read image {record.path}: {exc}") from exc
        if self.train:
            image = train_classification_image(image, self.size, random)
        else:
            image = eval_image(image, self.size)
        array = to_normalized_array(image, self.mean, self.std)
        label = 1.0 if record.is_ai else 0.0
        return torch.from_numpy(array), torch.tensor(label, dtype=torch.float32)


class LocalizationDataset(Dataset):
    def __init__(self, records: list[ImageRecord], size: int = 512, train: bool = False):
        self.records = [record for record in records if record.mask is not None]
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        try:
            image = load_rgb(record.path)
            with Image.open(record.mask) as handle:
                mask = handle.convert("L")
        except Exception as exc:  # pragma: no cover - filesystem dependent
            raise RuntimeError(f"Could not read sample {record.path} / {record.mask}: {exc}") from exc
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        if self.train:
            image, mask = train_localization_pair(image, mask, self.size, random)
        else:
            image = eval_image(image, self.size)
            mask = resize(mask, self.size, "nearest")
        array = to_normalized_array(image, IMAGENET_MEAN, IMAGENET_STD)
        mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        return torch.from_numpy(array), torch.from_numpy(mask_array).unsqueeze(0)


def label_balance(records: list[ImageRecord]) -> dict[str, int]:
    ai = sum(1 for record in records if record.is_ai)
    real = len(records) - ai
    return {"real": real, "ai": ai}
