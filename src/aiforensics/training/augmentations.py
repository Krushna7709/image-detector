"""Augmentations shared by training and robustness evaluation.

Every function operates on PIL images and accepts a ``random.Random``-like
object so calls are reproducible when a seeded generator is passed.
Classifications and masks receive the same geometric transforms.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    return image


def _resample(name: str = "bilinear") -> int:
    return {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "nearest": Image.Resampling.NEAREST,
    }[name]


def resize(image: Image.Image, size: int, resample: str = "bilinear") -> Image.Image:
    if image.width == size and image.height == size:
        return image
    return image.resize((size, size), _resample(resample))


def random_crop_resize(
    image: Image.Image,
    mask: Image.Image | None,
    size: int,
    rng: random.Random,
    scale: tuple[float, float] = (0.7, 1.0),
) -> tuple[Image.Image, Image.Image | None]:
    """Random-area crop resized to a square, keeping image and mask aligned."""

    width, height = image.size
    area = width * height * rng.uniform(*scale)
    ratio = rng.uniform(0.8, 1.25)
    crop_w = int(round(min(width, max(16, (area * ratio) ** 0.5))))
    crop_h = int(round(min(height, max(16, (area / ratio) ** 0.5))))
    if crop_w >= width and crop_h >= height:
        return resize(image, size), resize(mask, size, "nearest") if mask else None
    x0 = rng.randint(0, max(0, width - crop_w))
    y0 = rng.randint(0, max(0, height - crop_h))
    box = (x0, y0, x0 + crop_w, y0 + crop_h)
    cropped = image.crop(box)
    cropped_mask = mask.crop(box) if mask is not None else None
    return (
        resize(cropped, size),
        resize(cropped_mask, size, "nearest") if cropped_mask is not None else None,
    )


def random_flip(
    image: Image.Image,
    mask: Image.Image | None,
    rng: random.Random,
    p: float = 0.5,
) -> tuple[Image.Image, Image.Image | None]:
    if rng.random() >= p:
        return image, mask
    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mask is not None else None
    return image, mask


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return handle.convert("RGB")


def random_jpeg(image: Image.Image, rng: random.Random, p: float = 0.6) -> Image.Image:
    if rng.random() >= p:
        return image
    return jpeg_compress(image, rng.randint(30, 100))


def random_resize(image: Image.Image, rng: random.Random, p: float = 0.25) -> Image.Image:
    if rng.random() >= p:
        return image
    scale = rng.uniform(0.4, 0.9)
    small = image.resize(
        (max(16, int(image.width * scale)), max(16, int(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    return small.resize(image.size, Image.Resampling.BICUBIC)


def random_blur(image: Image.Image, rng: random.Random, p: float = 0.2) -> Image.Image:
    if rng.random() >= p:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.8)))


def random_noise(image: Image.Image, rng: random.Random, p: float = 0.2) -> Image.Image:
    if rng.random() >= p:
        return image
    sigma = rng.uniform(2.0, 10.0)
    array = np.asarray(image, dtype=np.float32) + np.random.default_rng(rng.randrange(2**32)).normal(
        0.0, sigma, np.asarray(image).shape
    )
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def train_classification_image(image: Image.Image, size: int, rng: random.Random) -> Image.Image:
    image, _ = random_crop_resize(image, None, size, rng)
    image, _ = random_flip(image, None, rng)
    image = random_jpeg(image, rng)
    image = random_resize(image, rng)
    image = random_blur(image, rng)
    image = random_noise(image, rng)
    return image


def train_localization_pair(
    image: Image.Image, mask: Image.Image, size: int, rng: random.Random
) -> tuple[Image.Image, Image.Image]:
    image, mask = random_crop_resize(image, mask, size, rng)
    assert mask is not None
    image, mask = random_flip(image, mask, rng)
    image = random_jpeg(image, rng)
    image = random_blur(image, rng)
    image = random_noise(image, rng)
    return image, mask


def eval_image(image: Image.Image, size: int, resample: str = "bilinear") -> Image.Image:
    return resize(image, size, resample)


def to_normalized_array(image: Image.Image, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean) / std
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def distort(image: Image.Image, name: str) -> Image.Image:
    """Apply one robustness distortion; matches the engine's stress settings."""

    name = name.lower()
    if name == "none":
        return image
    if name == "jpeg50":
        return jpeg_compress(image, 50)
    if name == "jpeg30":
        return jpeg_compress(image, 30)
    if name == "resize50":
        small = image.resize(
            (max(16, image.width // 2), max(16, image.height // 2)),
            Image.Resampling.BILINEAR,
        )
        return small.resize(image.size, Image.Resampling.BICUBIC)
    if name == "blur2":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    raise ValueError(f"Unknown distortion '{name}'")
