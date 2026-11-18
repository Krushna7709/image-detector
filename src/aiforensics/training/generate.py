"""Optional synthetic-data generation with diffusers.

Requires the optional ``diffusers`` package and model weights. Generated
images are untracked source data: respect each model's licence and never mix
them into the test split of a generator they resemble. This script only writes
images and a generation log; use ``aiforensics.training.manifest`` to index the
output directory.

    python -m aiforensics.training.generate \\
        --model stabilityai/stable-diffusion-xl-base-1.0 \\
        --out data/sdxl --count 64 --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PROMPTS = [
    "A weathered fisherman mending a net at sunrise, documentary photograph",
    "An empty library with oak shelves and soft window light, wide angle photograph",
    "A red bicycle leaning against a pastel wall in Lisbon, street photography",
    "A close-up portrait of an elderly woman with silver hair, studio lighting",
    "A mountain lake reflecting pine trees at dawn, landscape photograph",
    "A cozy kitchen with copper pots and a steaming kettle, natural light",
    "A futuristic train station with glass canopies at blue hour",
    "A bowl of ramen with soft-boiled egg on a dark wooden table, food photography",
    "A fox crossing a snowy forest clearing in winter, wildlife photograph",
    "An abstract paper sculpture in a white gallery, product photography",
    "A crowded night market lit by lanterns, candid street photograph",
    "A coral reef with tropical fish in clear turquoise water, underwater photo",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "image"


def generate_images(
    model: str,
    out_dir: Path,
    prompts: list[str],
    count: int,
    steps: int = 28,
    guidance: float = 6.0,
    seed: int = 1,
    width: int = 1024,
    height: int = 1024,
    device: str = "auto",
    dtype: str = "auto",
) -> Path:
    if not prompts:
        raise RuntimeError("At least one prompt is required.")
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "diffusers is not installed. Install the optional training extras, then retry."
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype_value = torch.float16 if device == "cuda" else torch.float32
    elif dtype == "fp16":
        dtype_value = torch.float16
    else:
        dtype_value = torch.float32
    if width % 8 or height % 8:
        raise RuntimeError("Width and height should be multiples of 8.")

    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Loading {model} on {device} ({dtype_value}).")
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=dtype_value)
    pipe = pipe.to(device)
    generator_device = "cuda" if device == "cuda" else "cpu"
    records = []
    for index in range(count):
        prompt = prompts[index % len(prompts)]
        image_seed = seed + index
        generator = torch.Generator(device=generator_device).manual_seed(image_seed)
        output = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        image = output.images[0]
        name = f"{_slug(model.rsplit('/', 1)[-1])}-{image_seed:08d}.png"
        image.save(destination / name)
        records.append(
            {
                "file": name,
                "prompt": prompt,
                "seed": image_seed,
                "steps": steps,
                "guidance": guidance,
                "width": width,
                "height": height,
            }
        )
        print(f"[{index + 1}/{count}] {name}")

    log_path = destination / "generation.json"
    log_path.write_text(
        json.dumps(
            {
                "model": model,
                "device": device,
                "dtype": str(dtype_value).replace("torch.", ""),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "images": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} images and {log_path}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic images with diffusers.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompts", type=Path, default=None, help="Text file with one prompt per line.")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp16", "fp32"), default="auto")
    args = parser.parse_args(argv)

    if args.prompts:
        prompts = [
            line.strip()
            for line in args.prompts.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        prompts = DEFAULT_PROMPTS
    try:
        generate_images(
            model=args.model,
            out_dir=args.out,
            prompts=prompts,
            count=max(1, args.count),
            steps=max(1, args.steps),
            guidance=args.guidance,
            seed=args.seed,
            width=args.width,
            height=args.height,
            device=args.device,
            dtype=args.dtype,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
