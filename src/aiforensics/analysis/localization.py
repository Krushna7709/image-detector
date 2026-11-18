"""Edit localization.

Classical mode combines residual-noise inconsistency with JPEG error-level
analysis to highlight regions that do not match the rest of the image. A
trained U-Net can be dropped in later without changing the API.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aiforensics.schemas import EditRegion


@dataclass
class LocalizationResult:
    heat: np.ndarray
    regions: list[EditRegion]
    edited_probability: float
    method: str
    note: str
    width: int
    height: int


def _relative_anomaly(values: np.ndarray, floor_ratio: float = 1.6, spread: float = 2.5) -> np.ndarray:
    """Highlight locally elevated values relative to the image's own median.

    Using a ratio instead of max-normalisation keeps a globally uniform image
    at zero, so "no local edit" stays a low probability instead of inventing
    peaks.
    """

    median = float(np.median(values))
    ratio = values / (median + 1e-9)
    return np.clip((ratio - floor_ratio) / spread, 0.0, 1.0)


def _extract_regions(heat: np.ndarray) -> list[EditRegion]:
    h, w = heat.shape
    threshold = max(0.62, float(np.percentile(heat, 98)))
    mask = (heat >= threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = max(24, int(0.002 * w * h))
    min_dim = max(10, int(0.01 * min(h, w)))
    candidates: list[tuple[int, EditRegion]] = []
    for idx in range(1, count):
        x, y, bw, bh, area = stats[idx]
        if area < min_area or bw < min_dim or bh < min_dim:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > 6.0:
            # Long thin components track texture and compression boundaries.
            continue
        score = float(heat[labels == idx].mean())
        candidates.append(
            (
                int(area),
                EditRegion(x=int(x), y=int(y), width=int(bw), height=int(bh), score=round(score, 4)),
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [region for _, region in candidates[:4]]


def summarize_heat(heat: np.ndarray) -> tuple[float, float, float]:
    """Return (top-1% mean, area ratio above 0.72, conservative probability)."""

    h, w = heat.shape
    top_values = np.sort(heat.ravel())[-max(1, h * w // 100):]
    top_mean = float(top_values.mean())
    area_ratio = float((heat > 0.72).mean())
    probability = float(np.clip(0.55 * top_mean + 0.9 * area_ratio, 0.0, 0.95))
    return top_mean, area_ratio, probability


def regions_from_heat(heat: np.ndarray, min_probability: float = 0.25) -> list[EditRegion]:
    _, _, probability = summarize_heat(heat)
    if probability < min_probability:
        return []
    return _extract_regions(heat)


def _compact_edit_probability(regions: list[EditRegion], total_pixels: int) -> float:
    """Edited regions are usually compact, so large smooth/textured areas do not count."""

    compact = []
    for region in regions:
        fraction = (region.width * region.height) / max(1, total_pixels)
        if 0.0008 <= fraction <= 0.15:
            compact.append((fraction, region))
    if not compact:
        return 0.08
    if len(compact) == 1 and compact[0][0] < 0.02:
        # A single tiny patch is usually an edge or compression artefact.
        return 0.08
    fraction, best = max(compact, key=lambda item: item[1].score)
    weight = 0.35 + 0.55 * min(1.0, fraction / 0.06)
    return float(np.clip(best.score * weight, 0.0, 0.65))


class HeuristicLocalizer:
    def localize(self, rgb_uint8: np.ndarray) -> LocalizationResult:
        rgb = rgb_uint8.astype(np.float32) / 255.0
        gray = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) * 255.0).astype(np.uint8)
        g = gray.astype(np.float32) / 255.0

        residual = np.abs(g - cv2.medianBlur(gray, 3).astype(np.float32) / 255.0)
        local_noise = cv2.GaussianBlur(residual, (0, 0), sigmaX=9)
        anomaly = _relative_anomaly(local_noise, floor_ratio=1.9, spread=2.2)

        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            ela = np.abs(g - decoded)
            ela_local = cv2.GaussianBlur(ela, (0, 0), sigmaX=8)
            ela_norm = _relative_anomaly(ela_local, floor_ratio=1.6, spread=2.0)
        else:
            ela_norm = np.zeros_like(anomaly)

        heat = 0.55 * anomaly + 0.45 * ela_norm
        heat = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), sigmaX=5)

        h, w = heat.shape
        regions = _extract_regions(heat)
        edited = _compact_edit_probability(regions, h * w)
        if edited < 0.25:
            regions = []

        note = (
            "Heuristic localization: highlights compact regions whose noise and recompression "
            "behaviour differ from the rest of the image. A single small patch is treated as "
            "inconclusive. It is not a trained edit mask."
        )
        return LocalizationResult(
            heat=heat,
            regions=regions,
            edited_probability=round(edited, 4),
            method="heuristic",
            note=note,
            width=w,
            height=h,
        )
