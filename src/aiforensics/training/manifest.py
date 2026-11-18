"""Dataset indexing, generator-disjoint splits and manifest tooling.

A manifest is a CSV file with these columns::

    path,label,generator,split,mask

``path`` may be absolute or relative to the manifest. ``label`` is ``real`` or
``ai``. ``generator`` names the model that produced an AI image (``real`` for
photographs). ``split`` is ``train``, ``val`` or ``test``. ``mask`` is optional
and points at a binary manipulation mask for localization training.

Run as a module to build a generator-disjoint manifest from directories::

    python -m aiforensics.training.manifest \\
        --out data/index.csv \\
        --real data/coco --real data/raise \\
        --ai sdxl=data/sdxl --ai flux=data/flux --ai sd15=data/sd15
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MANIFEST_COLUMNS = ("path", "label", "generator", "split", "mask")
SPLITS = ("train", "val", "test")
_LABEL_ALIASES = {"ai": "ai", "synthetic": "ai", "ai_generated": "ai", "fake": "ai", "real": "real"}


class ManifestError(ValueError):
    """Raised for malformed manifests and impossible split requests."""


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    generator: str
    split: str = ""
    mask: Path | None = None

    @property
    def is_ai(self) -> bool:
        return self.label == "ai"


def _normalise_label(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value not in _LABEL_ALIASES:
        raise ManifestError(f"Unknown label '{raw}'. Use 'ai' or 'real'.")
    return _LABEL_ALIASES[value]


def read_manifest(path: Path) -> list[ImageRecord]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise ManifestError(f"Manifest not found: {manifest_path}")
    root = manifest_path.parent
    records: list[ImageRecord] = []
    seen: set[Path] = set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ManifestError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            if not raw_path:
                continue
            image_path = Path(raw_path).expanduser()
            if not image_path.is_absolute():
                image_path = (root / image_path).resolve()
            if image_path in seen:
                continue
            seen.add(image_path)
            label = _normalise_label(row.get("label") or "")
            generator = (row.get("generator") or ("real" if label == "real" else "")).strip()
            if label == "ai" and not generator:
                raise ManifestError(f"Line {line_number}: AI images need a generator name.")
            split = (row.get("split") or "").strip().lower()
            if split and split not in SPLITS:
                raise ManifestError(f"Line {line_number}: unknown split '{split}'.")
            mask_raw = (row.get("mask") or "").strip()
            mask = None
            if mask_raw:
                mask = Path(mask_raw).expanduser()
                if not mask.is_absolute():
                    mask = (root / mask).resolve()
            records.append(
                ImageRecord(
                    path=image_path,
                    label=label,
                    generator=generator or "real",
                    split=split,
                    mask=mask,
                )
            )
    if not records:
        raise ManifestError(f"Manifest {manifest_path} contains no rows.")
    return records


def _format_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return os.path.relpath(path.resolve(), root.resolve()).replace("\\", "/")


def write_manifest(records: list[ImageRecord], path: Path) -> Path:
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    root = manifest_path.parent
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": _format_path(record.path, root),
                    "label": record.label,
                    "generator": record.generator,
                    "split": record.split,
                    "mask": _format_path(record.mask, root) if record.mask else "",
                }
            )
    return manifest_path


def collect_images(directory: Path) -> list[Path]:
    base = Path(directory).expanduser().resolve()
    if not base.exists():
        raise ManifestError(f"Directory not found: {base}")
    return sorted(
        path for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def assign_generator_disjoint_splits(
    records: list[ImageRecord],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 13,
) -> list[ImageRecord]:
    """Assign whole AI generators to splits so test generators are unseen.

    Generator-disjoint splits prevent a detector from learning a generator's
    fingerprint instead of the shared synthetic signature. Real photographs
    are split randomly.
    """

    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ManifestError("Split fractions must be between 0 and 1.")
    if val_fraction + test_fraction >= 1:
        raise ManifestError("val_fraction + test_fraction must be below 1.")

    rng = random.Random(seed)
    ai_records = [record for record in records if record.is_ai]
    real_records = [record for record in records if not record.is_ai]
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in ai_records:
        groups[record.generator].append(record)
    if len(groups) < 3:
        raise ManifestError(
            "Generator-disjoint splits need at least three AI generators, "
            f"found {len(groups)}: {', '.join(sorted(groups)) or 'none'}. "
            "Add more generators or build the manifest with --no-split."
        )

    names = sorted(groups)
    rng.shuffle(names)
    names.sort(key=lambda name: len(groups[name]), reverse=True)
    n_groups = len(names)
    n_val = max(1, round(n_groups * val_fraction))
    n_test = max(1, round(n_groups * test_fraction))
    while n_val + n_test >= n_groups:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ManifestError(
                "Not enough generators to populate train, val and test. "
                "Add more generators or use --no-split."
            )
    n_train = n_groups - n_val - n_test
    assigned = {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }

    group_split = {
        name: split for split, names_in_split in assigned.items() for name in names_in_split
    }
    updated = [replace(record, split=group_split[record.generator]) for record in ai_records]

    shuffled_real = list(real_records)
    rng.shuffle(shuffled_real)
    total = len(shuffled_real)
    n_test = round(total * test_fraction)
    n_val = round(total * val_fraction)
    n_train = total - n_test - n_val
    for index, record in enumerate(shuffled_real):
        if index < n_train:
            split = "train"
        elif index < n_train + n_val:
            split = "val"
        else:
            split = "test"
        updated.append(replace(record, split=split))
    return updated


def resolve_splits(
    records: list[ImageRecord],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 13,
) -> list[ImageRecord]:
    """Return records with splits, creating a split when none exist.

    Three or more AI generators use the same generator-disjoint allocation as
    ``build_manifest``. Smaller experiments fall back to a label-stratified
    random split so both classes remain in training.
    """

    if any(record.split for record in records):
        if not any(record.split == "train" for record in records):
            raise ManifestError("Manifest defines splits but none are 'train'.")
        return records
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ManifestError("Split fractions must be between 0 and 1.")
    if val_fraction + test_fraction >= 1:
        raise ManifestError("val_fraction + test_fraction must be below 1.")

    generators = {record.generator for record in records if record.is_ai}
    if len(generators) >= 3:
        return assign_generator_disjoint_splits(
            records,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    rng = random.Random(seed)
    split_by_path: dict[Path, str] = {}
    for label in ("real", "ai"):
        items = [record for record in records if record.label == label]
        rng.shuffle(items)
        total = len(items)
        n_val = n_test = 0
        if total >= 5:
            n_test = max(1, round(total * test_fraction))
            n_val = max(1, round(total * val_fraction))
            while n_val + n_test >= total:
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
        n_train = total - n_val - n_test
        for index, record in enumerate(items):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_by_path[record.path] = split
    return [replace(record, split=split_by_path[record.path]) for record in records]


def split_records(records: list[ImageRecord], split: str) -> list[ImageRecord]:
    return [record for record in records if record.split == split]


def summarize(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    generators: dict[str, set[str]] = {split: set() for split in SPLITS}
    for record in records:
        split = record.split or "unassigned"
        table.setdefault(split, Counter())[record.label] += 1
        generators.setdefault(split, set()).add(record.generator)
    return {
        split: {
            "total": sum(counter.values()),
            "real": counter.get("real", 0),
            "ai": counter.get("ai", 0),
            "generators": len(generators[split]),
        }
        for split, counter in table.items()
    }


def build_manifest(
    real_dirs: list[Path],
    ai_dirs: dict[str, Path],
    out: Path,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 13,
    split: bool = True,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen: set[Path] = set()
    for directory in real_dirs:
        for path in collect_images(directory):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(ImageRecord(path=resolved, label="real", generator="real"))
    for generator, directory in ai_dirs.items():
        for path in collect_images(directory):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(ImageRecord(path=resolved, label="ai", generator=generator))
    if not any(record.is_ai for record in records):
        raise ManifestError("No AI images were found; provide at least one --ai NAME=DIR.")
    if not any(not record.is_ai for record in records):
        raise ManifestError("No real images were found; provide at least one --real DIR.")
    if split:
        records = assign_generator_disjoint_splits(
            records, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
        )
    write_manifest(records, out)
    return records


def _parse_ai_args(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ManifestError(f"Expected NAME=DIR for --ai, got '{value}'.")
        name, directory = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ManifestError("AI generator names cannot be empty.")
        if name in result:
            raise ManifestError(f"Duplicate AI generator name '{name}'.")
        result[name] = Path(directory)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a generator-disjoint dataset manifest.")
    parser.add_argument("--out", required=True, type=Path, help="Destination CSV path.")
    parser.add_argument("--real", action="append", default=[], type=Path, help="Directory of real photos (repeatable).")
    parser.add_argument("--ai", action="append", default=[], help="NAME=DIR pair for AI images (repeatable).")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-split", action="store_true", help="Only index files; do not assign splits.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        print(f"Refusing to overwrite {args.out}; pass --force to replace it.", file=sys.stderr)
        return 2
    if not args.real:
        print("Provide at least one --real directory.", file=sys.stderr)
        return 2
    try:
        records = build_manifest(
            real_dirs=args.real,
            ai_dirs=_parse_ai_args(args.ai),
            out=args.out,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            split=not args.no_split,
        )
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = summarize(records)
    print(f"Wrote {len(records)} records to {args.out}")
    for split, stats in summary.items():
        generators = stats["generators"]
        print(
            f"  {split:10s} total={stats['total']:5d} real={stats['real']:5d} "
            f"ai={stats['ai']:5d} generators={generators}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
