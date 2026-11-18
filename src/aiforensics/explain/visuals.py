"""Rendering helpers: data URLs, heatmap overlays, region boxes, spectra."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from aiforensics.schemas import EditRegion

_ACCENT_BGR = (177, 224, 70)  # mint (#46e0b1)
_TEXT_BGR = (241, 237, 230)
_MUTED_BGR = (150, 165, 175)


def encode_png(arr: np.ndarray) -> str:
    """Encode an RGB/RGBA uint8 array as a data URL."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = arr
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def heatmap_overlay(base_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a JET heatmap over the image, weighted by local heat strength.

    Weighting the alpha per pixel keeps weak, diffuse variation from tinting
    the whole photograph, so only elevated regions stand out.
    """
    h, w = base_rgb.shape[:2]
    if heat.shape[:2] != (h, w):
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = np.clip(heat, 0.0, 1.0)
    strength = np.clip((heat - 0.06) / 0.94, 0.0, 1.0) ** 0.75
    bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    colored = cv2.applyColorMap((strength * 255).astype(np.uint8), cv2.COLORMAP_JET)
    weight = (alpha * strength)[..., None]
    blended = bgr.astype(np.float32) * (1.0 - weight) + colored.astype(np.float32) * weight
    return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def regions_overlay(shape: tuple[int, int], regions: list[EditRegion]) -> np.ndarray:
    """Transparent RGBA layer with bounding boxes and scores."""
    h, w = shape
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    thickness = max(2, round(min(h, w) / 280))
    font_scale = max(0.4, min(h, w) / 900)
    for region in regions:
        x, y = max(0, region.x), max(0, region.y)
        x2 = min(w - 1, region.x + region.width)
        y2 = min(h - 1, region.y + region.height)
        color = (84, 180, 255, 255)  # BGR + alpha
        cv2.rectangle(canvas, (x, y), (x2, y2), color, thickness)
        label = f"{region.score:.0%}"
        ty = y - 8 if y > 20 else y2 + 18
        cv2.putText(
            canvas, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, max(1, thickness - 1), cv2.LINE_AA,
        )
    return canvas


def _radial_profile(power: np.ndarray, n_bins: int = 96) -> np.ndarray:
    h, w = power.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt(((x - cx) / w) ** 2 + ((y - cy) / h) ** 2)
    bins = np.clip((radius / 0.5 * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)
    sums = np.bincount(bins.ravel(), weights=power.ravel(), minlength=n_bins)
    counts = np.bincount(bins.ravel(), minlength=n_bins)
    return (sums / np.maximum(counts, 1)).astype(np.float64)


def spectrum_data_url(rgb: np.ndarray, size: int = 260) -> str:
    """Frequency spectrum + radial power profile, rendered with OpenCV."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * window))
    power = np.abs(spectrum) ** 2
    magnitude = np.log1p(np.abs(spectrum))
    magnitude = (magnitude - magnitude.min()) / (np.ptp(magnitude) + 1e-9)
    spec_bgr = cv2.applyColorMap((magnitude * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    canvas = np.full((size, size + 220, 3), (18, 13, 9), dtype=np.uint8)
    canvas[:, :size] = spec_bgr
    cv2.line(canvas, (size, 0), (size, size), (48, 40, 32), 1)

    profile = _radial_profile(power)
    logp = np.log10(profile + 1e-12)
    logp = (logp - logp.min()) / (np.ptp(logp) + 1e-9)
    x0, y0, plot_w, plot_h = size + 26, 26, 168, size - 58
    cv2.rectangle(canvas, (x0, y0), (x0 + plot_w, y0 + plot_h), (45, 38, 32), 1)
    pts = np.array(
        [
            (x0 + int(i / (len(logp) - 1) * plot_w), y0 + plot_h - int(v * plot_h))
            for i, v in enumerate(logp)
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [pts], False, _ACCENT_BGR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "RADIAL POWER", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _MUTED_BGR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "low", (x0, y0 + plot_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, _MUTED_BGR, 1, cv2.LINE_AA)
    label = "high"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
    cv2.putText(canvas, label, (x0 + plot_w - tw, y0 + plot_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, _MUTED_BGR, 1, cv2.LINE_AA)

    return encode_png(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def gradcam_overlay(base_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return heatmap_overlay(base_rgb, cam, alpha=alpha)
