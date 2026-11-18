"""Synthesize local edits plus binary masks for localizer training.

This creates copy-paste, blur, colour and noise edits on source photographs.
The masks mark exactly the manipulated rectangles, giving the U-Net a
supervised localization task without hunting for pixel-level ground truth.

    python -m aiforensics.training.make_edits \\
        --source-manifest data/index.csv --real-only \\
        --images-out data/edits/images --masks-out data/edits/masks \\
        --out-manifest data/edits/index.csv --per-image 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from aiforensics.training.augmentations import jpeg_compress, load_rgb
from aiforensics.training.manifest import (
    ImageRecord,
    ManifestError,
    read_manifest,
    resolve_splits,
    write_manifest,
)

EDIT_TYPES = ("paste", "blur", "colour", "noise")


def _random_box(width: int, height: int, rng: random.Random) -> tuple[int, int, int, int]:
    target_area = width * height * rng.uniform(0.04, 0.18)
    ratio = rng.uniform(0.6, 1.7)
    box_w = int(round(min(width - 4, max(16, (target_area * ratio) ** 0.5))))
    box_h = int(round(min(height - 4, max(16, (target_area / ratio) ** 0.5))))
    x0 = rng.randint(0, max(0, width - box_w))
    y0 = rng.randint(0, max(0, height - box_h))
    return x0, y0, x0 + box_w, y0 + box_h


def _apply_edit(
    image: Image.Image, box: tuple[int, int, int, int], edit: str, rng: random.Random
) -> None:
    x0, y0, x1, y1 = box
    region = image.crop(box)
    if edit == "paste":
        width, height = image.size
        donor_w, donor_h = x1 - x0, y1 - y0
        dx = rng.randint(0, max(0, width - donor_w))
        dy = rng.randint(0, max(0, height - donor_h))
        donor = image.crop((dx, dy, dx + donor_w, dy + donor_h))
        image.paste(donor, (x0, y0))
    elif edit == "blur":
        image.paste(region.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.0, 6.0))), (x0, y0))
    elif edit == "colour":
        array = np.asarray(region, dtype=np.float32)
        gains = np.asarray([rng.uniform(0.75, 1.25) for _ in range(3)], dtype=np.float32)
        offsets = np.asarray([rng.uniform(-30, 30) for _ in range(3)], dtype=np.float32)
        array = np.clip(array * gains + offsets, 0, 255).astype(np.uint8)
        image.paste(Image.fromarray(array, "RGB"), (x0, y0))
    elif edit == "noise":
        array = np.asarray(region, dtype=np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(
            0.0, rng.uniform(8.0, 22.0), array.shape
        )
        image.paste(Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB"), (x0, y0))
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unknown edit type '{edit}'.")


def make_edits(
    source_manifest: Path,
    images_out: Path,
    masks_out: Path,
    out_manifest: Path,
    per_image: int = 2,
    seed: int = 13,
    real_only: bool = False,
    jpeg_probability: float = 0.5,
    max_images: int | None = None,
) -> list[ImageRecord]:
    records = read_manifest(source_manifest)
    if not any(record.split for record in records):
        records = resolve_splits(records, seed=seed)
    if real_only:
        records = [record for record in records if not record.is_ai]
    if max_images is not None:
        records = records[:max_images]
    if not records:
        raise RuntimeError("No source images selected; check the manifest/filters.")

    images_dir = Path(images_out).expanduser().resolve()
    masks_dir = Path(masks_out).expanduser().resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    output: list[ImageRecord] = []

    for record in records:
        try:
            base = load_rgb(record.path)
        except Exception as exc:
            print(f"skip {record.path}: {exc}", file=sys.stderr)
            continue
        if min(base.size) < 64:
            print(f"skip {record.path}: image is smaller than 64 px", file=sys.stderr)
            continue
        for index in range(per_image):
            image = base.copy()
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            edit_count = 2 if rng.random() < 0.5 else 1
            for _ in range(edit_count):
                box = _random_box(image.width, image.height, rng)
                edit = rng.choice(EDIT_TYPES)
                _apply_edit(image, box, edit, rng)
                draw.rectangle(box, fill=255)
            if rng.random() < jpeg_probability:
                image = jpeg_compress(image, rng.randint(55, 90))
            stem = record.path.stem[:80]
            name = f"{stem}-edit{index:02d}-{rng.randrange(2**24):06x}.png"
            image_path = images_dir / name
            mask_path = masks_dir / name
            image.save(image_path, format="PNG")
            mask.save(mask_path, format="PNG")
            output.append(
                ImageRecord(
                    path=image_path,
                    label=record.label,
                    generator=record.generator,
                    split=record.split,
                    mask=mask_path,
                )
            )

    if not output:
        raise RuntimeError("No edits were produced.")
    write_manifest(output, out_manifest)
    print(f"Wrote {len(output)} edited images and masks; manifest: {out_manifest}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize edits and masks for localizer training.")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--images-out", required=True, type=Path)
    parser.add_argument("--masks-out", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--real-only", action="store_true", help="Only edit real photographs.")
    parser.add_argument("--jpeg-probability", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        make_edits(
            source_manifest=args.source_manifest,
            images_out=args.images_out,
            masks_out=args.masks_out,
            out_manifest=args.out_manifest,
            per_image=max(1, args.per_image),
            seed=args.seed,
            real_only=args.real_only,
            jpeg_probability=min(1.0, max(0.0, args.jpeg_probability)),
            max_images=args.max_images,
        )
    except (RuntimeError, ManifestError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
