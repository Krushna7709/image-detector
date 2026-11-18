"""Image loading, validation and distortion simulation."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from aiforensics.config import Settings
from aiforensics.config import settings as default_settings


class ImageValidationError(ValueError):
    """Raised when an upload cannot be analysed."""

    def __init__(self, message: str, status_code: int = 400, code: str = "invalid_image"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _pixel_limit_message(settings: Settings) -> str:
    megapixels = settings.max_pixels / 1_000_000
    return f"Image resolution is too large (limit: {megapixels:g} megapixels)."


def _decode_checked(data: bytes, settings: Settings) -> tuple[Image.Image, str]:
    """Open and decode an image only after validating its format and dimensions."""

    try:
        pil = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            _pixel_limit_message(settings), 413, "too_many_pixels"
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(
            "Unsupported or corrupted image file.", 415, "unsupported_format"
        ) from exc

    fmt = (pil.format or "").upper()
    if fmt not in settings.allowed_formats:
        raise ImageValidationError(
            f"Unsupported format '{fmt or 'unknown'}'. Use JPEG, PNG, WEBP, BMP or TIFF.",
            415,
            "unsupported_format",
        )
    width, height = pil.size
    if width * height > settings.max_pixels:
        raise ImageValidationError(
            _pixel_limit_message(settings), 413, "too_many_pixels"
        )

    try:
        pil.load()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            _pixel_limit_message(settings), 413, "too_many_pixels"
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(
            "Unsupported or corrupted image file.", 415, "unsupported_format"
        ) from exc
    return pil, fmt


@dataclass
class LoadedImage:
    filename: str
    sha256: str
    file_size: int
    format: str
    width: int
    height: int
    pil: Image.Image
    rgb: np.ndarray
    analysis_rgb: np.ndarray
    analysis: np.ndarray
    display_rgb: np.ndarray


def _resize_side(pil: Image.Image, max_side: int) -> Image.Image:
    w, h = pil.size
    longest = max(w, h)
    if longest <= max_side:
        return pil
    scale = max_side / longest
    return pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def load_image(data: bytes, filename: str, settings: Settings) -> LoadedImage:
    if not data:
        raise ImageValidationError("The uploaded file is empty.", code="empty_file")
    if len(data) > settings.max_upload_bytes:
        raise ImageValidationError(
            f"File is larger than the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
            status_code=413,
            code="file_too_large",
        )

    pil, fmt = _decode_checked(data, settings)
    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:  # malformed EXIF should never block analysis
        pass
    pil = pil.convert("RGB")
    width, height = pil.size

    analysis_pil = _resize_side(pil, settings.analysis_max_side)
    display_pil = _resize_side(pil, settings.display_max_side)

    rgb = np.asarray(pil, dtype=np.uint8)
    analysis_rgb = np.asarray(analysis_pil, dtype=np.uint8)
    display_rgb = np.asarray(display_pil, dtype=np.uint8)
    analysis = analysis_rgb.astype(np.float32) / 255.0

    return LoadedImage(
        filename=filename,
        sha256=hashlib.sha256(data).hexdigest(),
        file_size=len(data),
        format=fmt,
        width=width,
        height=height,
        pil=pil,
        rgb=rgb,
        analysis_rgb=analysis_rgb,
        analysis=analysis,
        display_rgb=display_rgb,
    )


def distort_image(
    data: bytes, name: str, settings: Settings | None = None
) -> tuple[bytes, str]:
    """Simulate real-world image handling and return new file bytes + label."""

    if name in ("", "none"):
        return data, "none"
    if name not in ("jpeg50", "jpeg30", "resize50", "blur2"):
        raise ImageValidationError(f"Unknown distortion '{name}'.", 400, "invalid_distortion")

    config = settings or default_settings
    pil, _ = _decode_checked(data, config)
    try:
        im = ImageOps.exif_transpose(pil).convert("RGB")
        buf = io.BytesIO()
        if name in ("jpeg50", "jpeg30"):
            quality = 50 if name == "jpeg50" else 30
            im.save(buf, format="JPEG", quality=quality)
            label = f"JPEG q{quality}"
        elif name == "resize50":
            w, h = im.size
            small = im.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)
            im = small.resize((w, h), Image.Resampling.BICUBIC)
            im.save(buf, format="PNG")
            label = "Resized 50% and restored"
        else:
            from PIL import ImageFilter

            im.filter(ImageFilter.GaussianBlur(radius=2.0)).save(buf, format="PNG")
            label = "Gaussian blur (sigma 2)"
    except (OSError, ValueError) as exc:
        raise ImageValidationError(
            "Cannot apply distortion to this file.", 400, "invalid_image"
        ) from exc

    return buf.getvalue(), label
