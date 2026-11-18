"""Classical forensic analysis.

This is the always-available evidence engine: it works with no downloads, no
GPU and no training data. It is deliberately conservative. Signals are
heuristic indicators, not calibrated probabilities, and the thresholds were
chosen from a small reference set plus published forensic observations:

* generated images tend to have a spatially uniform noise residual, while
  camera captures mix textured and smooth regions (and sometimes sensor noise);
* diffusion renders often lack a physical noise floor and have distributions of
  detail that differ from photographs;
* provenance metadata is the strongest single lead when it survives.

The engine is never the final word; the UI labels it as a fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from aiforensics.schemas import Signal

# Operating points for 8-bit photo-like images. These are indicators, not
# decision boundaries; the fusion step combines them. The noise-field and
# detail cues were calibrated on the bundled demo set and hold across JPEG
# recompression, resizing and blur; the remaining cues are weak tie-breakers.
NOISE_HET_CENTER = 2.30     # log10(p95/p50) of local noise variance; lower leans synthetic
NOISE_HET_SCALE = 0.35
DETAIL_FLAT_CENTER = 0.30   # share of near-flat pixels
DETAIL_FLAT_SCALE = 0.12
RESIDUAL_PIVOT = 0.028      # median-filter residual standard deviation
RESIDUAL_SPREAD = 0.018
SLOPE_CENTER = -2.6         # radial log-log power slope
SLOPE_SPREAD = 1.3
HF_PIVOT = 0.020            # energy above half-Nyquist
SAT_PIVOT = 0.42
COLOR_PIVOT = 45.0

# Generator decoders upsample on the pixel lattice, leaving a 2x2-periodic
# trace in the noise residual. Camera pixels (including JPEG output) do not
# show this periodicity; it fades when an image is resampled or recompressed.
GRID_CENTER = -2.05         # log10 of the 2x2 parity amplitude in the residual
GRID_SCALE = 0.45

# Grain measured in the smoothest regions of the frame. Camera pipelines and
# JPEG recompression leave a measurable noise floor there; clean renders do
# not. When grain is present, a spatially uniform residual is acquisition
# noise, not a generator artifact, so the noise-field cue is discounted.
GRAIN_CENTER = 0.0030
GRAIN_SCALE = 0.0005
GRAIN_GATE = 0.0032
GRAIN_NOISE_CAP = 0.25

# Heavy blur erases the noise floor and high-frequency detail that most of the
# evidence depends on. When it is detected, those cues are down-weighted and
# the fused probability is pulled toward the uncertain band instead of
# reporting a confident verdict from destroyed evidence.
BLUR_SLOPE = -5.5
BLUR_HF_RATIO = 0.0015
BLUR_CUE_RELIABILITY = 0.35
BLUR_ATTENUATION = 0.45
BLUR_RELIABILITY = 0.60

FUSION_CENTER = 0.442
FUSION_SCALE = 7.0

# A compact region whose noise and recompression behaviour differs from the
# rest of the frame can be an inpainted/composited area (for example an AI
# edited face). The cue is only trusted on lossless or very high-quality
# uploads: JPEG below ~q95 spreads enough block noise to mimic it.
LOCAL_EDIT_CENTER = 0.50
LOCAL_EDIT_SCALE = 0.09
LOCAL_EDIT_STRONG = 0.58
LOCAL_EDIT_MODERATE = 0.48
LOCAL_EDIT_FLOOR = 0.68
LOCAL_EDIT_UNCERTAIN_FLOOR = 0.54

WEIGHTS = {
    "noise_heterogeneity": 0.405,
    "detail_distribution": 0.166,
    "pixel_grid": 0.188,
    "noise_floor": 0.034,
    "spectral_decay": 0.034,
    "high_frequency": 0.011,
    "color_signature": 0.023,
    "provenance": 0.099,
    "local_edit": 0.040,
}

LABELS = {
    "noise_heterogeneity": "Noise field heterogeneity",
    "detail_distribution": "Detail distribution",
    "pixel_grid": "Pixel-grid periodicity",
    "noise_floor": "Residual noise floor",
    "spectral_decay": "Spectral decay",
    "high_frequency": "High-frequency balance",
    "color_signature": "Color signature",
    "provenance": "Provenance metadata",
    "local_edit": "Local edit consistency",
}


@dataclass
class HeuristicResult:
    probability: float
    reliability: float
    signals: list[Signal]
    features: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HeuristicAnalyzer:
    """Extracts complementary forensic signals from a single image."""

    def analyze(
        self,
        rgb: np.ndarray,
        *,
        generator_hint: str | None,
        has_camera_exif: bool,
        jpeg_quality: int | None,
        synthetic_provenance: bool = False,
        local_edit: float | None = None,
        local_edit_regions: int = 0,
    ) -> HeuristicResult:
        rgb = rgb.astype(np.float32) / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        features = self._features(gray, rgb)

        compressed = jpeg_quality is not None and jpeg_quality < 60
        freq_rel = 0.80 if compressed else 1.0
        blurred = (
            features["spectral_slope"] < BLUR_SLOPE
            or features["hf_ratio"] < BLUR_HF_RATIO
        )
        cue_rel = BLUR_CUE_RELIABILITY if blurred else 1.0

        noise_score = _sigmoid(
            (NOISE_HET_CENTER - features["noise_heterogeneity"]) / NOISE_HET_SCALE
        )
        grain_gated = features["smooth_grain"] >= GRAIN_GATE
        if grain_gated:
            noise_score = min(noise_score, GRAIN_NOISE_CAP)
        flat_score = _sigmoid(
            (DETAIL_FLAT_CENTER - features["flat_fraction"]) / DETAIL_FLAT_SCALE
        )
        grid_score = _sigmoid(
            (math.log10(max(features["pixel_grid"], 1e-6)) + 2.05) / GRID_SCALE
        )
        residual_score = _clip01(
            (RESIDUAL_PIVOT - features["residual_std"]) / RESIDUAL_SPREAD
        )
        slope_score = _clip01(abs(features["spectral_slope"] - SLOPE_CENTER) / SLOPE_SPREAD)
        hf_score = _clip01(1.0 - features["hf_ratio"] / HF_PIVOT)
        color_score = 0.6 * _clip01((features["saturation"] - SAT_PIVOT) / 0.25) + 0.4 * _clip01(
            (features["colorfulness"] - COLOR_PIVOT) / 50.0
        )

        if generator_hint:
            prov_score, prov_rel, prov_detail = 0.97, 0.90, f"Metadata names '{generator_hint}' (spoofable)."
        elif synthetic_provenance:
            prov_score, prov_rel, prov_detail = 0.90, 0.85, "C2PA claims trained algorithmic media."
        elif has_camera_exif:
            prov_score, prov_rel, prov_detail = 0.12, 0.55, "Camera make/model present in EXIF."
        else:
            prov_score, prov_rel, prov_detail = 0.55, 0.20, "No camera metadata; weak evidence either way."

        if local_edit is None:
            local_score, local_rel = 0.0, 0.05
            local_detail = (
                "Local edit check skipped: JPEG recompression below q95 spreads block noise "
                "that mimics local inconsistencies, so the cue is not trusted."
            )
        else:
            local_score = _sigmoid((local_edit - LOCAL_EDIT_CENTER) / LOCAL_EDIT_SCALE)
            local_rel = 0.55
            local_detail = (
                f"Compact local inconsistency probability {local_edit:.2f} across "
                f"{local_edit_regions} region(s). Noise and recompression mismatch is consistent "
                "with an inpainted or composited area (for example an edited face); ordinary "
                "retouching can trigger it too."
            )

        signals = [
            Signal(
                name="noise_heterogeneity",
                label=LABELS["noise_heterogeneity"],
                score=round(noise_score, 4),
                weight=WEIGHTS["noise_heterogeneity"],
                reliability=round(0.80 * cue_rel, 4),
                detail=(
                    f"Local noise variance spread p95/p50 = "
                    f"{10 ** features['noise_heterogeneity']:.1f} "
                    f"(log10 {features['noise_heterogeneity']:.2f}). Uniform fields lean synthetic."
                ),
            ),
            Signal(
                name="detail_distribution",
                label=LABELS["detail_distribution"],
                score=round(flat_score, 4),
                weight=WEIGHTS["detail_distribution"],
                reliability=0.85,
                detail=(
                    f"Near-flat pixel share {features['flat_fraction']:.0%} "
                    f"(reference centre {DETAIL_FLAT_CENTER:.0%}). "
                    "Renders tend to carry detail everywhere; photos mix texture and calm areas."
                ),
            ),
            Signal(
                name="pixel_grid",
                label=LABELS["pixel_grid"],
                score=round(grid_score, 4),
                weight=WEIGHTS["pixel_grid"],
                reliability=0.75,
                detail=(
                    f"2x2 noise-lattice periodicity {features['pixel_grid']:.4f} "
                    f"(log10 {math.log10(max(features['pixel_grid'], 1e-6)):.2f}). "
                    "Generator upsampling leaves a pixel-grid trace; camera and JPEG noise does not."
                ),
            ),
            Signal(
                name="noise_floor",
                label=LABELS["noise_floor"],
                score=round(residual_score, 4),
                weight=WEIGHTS["noise_floor"],
                reliability=round(0.55 * cue_rel, 4),
                detail=(
                    f"Median-filter residual std {features['residual_std']:.4f}. "
                    "Weakly informative alone; compression and blur reshape this floor."
                ),
            ),
            Signal(
                name="spectral_decay",
                label=LABELS["spectral_decay"],
                score=round(slope_score, 4),
                weight=WEIGHTS["spectral_decay"],
                reliability=round(0.40 * freq_rel * cue_rel, 4),
                detail=(
                    f"Radial log-log slope {features['spectral_slope']:.2f} "
                    f"(reference centre {SLOPE_CENTER:.1f})."
                ),
            ),
            Signal(
                name="high_frequency",
                label=LABELS["high_frequency"],
                score=round(hf_score, 4),
                weight=WEIGHTS["high_frequency"],
                reliability=round(0.30 * freq_rel * cue_rel, 4),
                detail=f"Energy above half-Nyquist: {features['hf_ratio']:.3f}. Weakly informative alone.",
            ),
            Signal(
                name="color_signature",
                label=LABELS["color_signature"],
                score=round(color_score, 4),
                weight=WEIGHTS["color_signature"],
                reliability=0.15,
                detail=(
                    f"Colorfulness {features['colorfulness']:.1f}, "
                    f"mean saturation {features['saturation']:.2f}. "
                    "Weakly informative alone; colour is content, not provenance."
                ),
            ),
            Signal(
                name="local_edit",
                label=LABELS["local_edit"],
                score=round(local_score, 4),
                weight=WEIGHTS["local_edit"],
                reliability=local_rel,
                detail=local_detail,
            ),
            Signal(
                name="provenance",
                label=LABELS["provenance"],
                score=prov_score,
                weight=WEIGHTS["provenance"],
                reliability=prov_rel,
                detail=prov_detail,
            ),
        ]

        numerator = sum(s.weight * s.reliability * s.score for s in signals)
        denominator = sum(s.weight * s.reliability for s in signals)
        weighted = numerator / max(denominator, 1e-9)
        reliability = sum(s.weight * s.reliability for s in signals) / sum(WEIGHTS.values())
        probability = _sigmoid((weighted - FUSION_CENTER) * FUSION_SCALE)

        notes: list[str] = []
        if blurred:
            probability = 0.5 + (probability - 0.5) * BLUR_ATTENUATION
            reliability *= BLUR_RELIABILITY
            notes.append(
                "Heavy blur or resampling detected; the noise field and high-frequency detail "
                "needed for a confident call are largely gone, so the estimate is pulled toward "
                "the uncertain band."
            )
        if generator_hint or synthetic_provenance:
            probability = max(probability, 0.90)
            if generator_hint:
                notes.append(
                    f"Provenance metadata names '{generator_hint}'; the probability is floored at 0.90 "
                    "because generator metadata is difficult to produce accidentally."
                )
            else:
                notes.append(
                    "Provenance metadata claims synthetic origin (C2PA trainedAlgorithmicMedia); "
                    "the probability is floored at 0.90."
                )
        if compressed:
            notes.append("JPEG compression detected; frequency-domain signals are down-weighted.")
        if local_edit is not None and not blurred and local_edit >= LOCAL_EDIT_STRONG:
            probability = max(probability, LOCAL_EDIT_FLOOR)
            notes.append(
                "A compact region shows noise and recompression behaviour that differs from the "
                "rest of the frame, consistent with an AI-inpainted area such as an edited face. "
                "The probability is raised into the AI band; ordinary retouching can trigger this "
                "cue, so inspect the highlighted region."
            )
        elif local_edit is not None and not blurred and local_edit >= LOCAL_EDIT_MODERATE:
            probability = max(probability, LOCAL_EDIT_UNCERTAIN_FLOOR)
            notes.append(
                "A compact local inconsistency was found but is not strong enough for a synthetic "
                "verdict; the estimate is raised to the uncertain band and needs review."
            )
        if grain_gated:
            notes.append(
                "Measurable grain in the smoothest regions suggests the uniform noise field is "
                "acquisition or compression noise, not a generator artifact, so that cue is discounted."
            )
        elif features["noise_heterogeneity"] < 1.5:
            notes.append(
                "The noise residual is unusually uniform across the frame; this is a known "
                "generator artifact, but denoised or synthetic-looking photos can show it too."
            )

        return HeuristicResult(
            probability=round(probability, 4),
            reliability=round(_clip01(reliability), 4),
            signals=signals,
            features=features,
            notes=notes,
        )

    @staticmethod
    def _features(gray: np.ndarray, rgb: np.ndarray) -> dict[str, float]:
        h, w = gray.shape
        scale = 512.0 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(
                gray,
                (max(8, round(w * scale)), max(8, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = gray

        sh, sw = small.shape
        window = np.outer(np.hanning(sh), np.hanning(sw)).astype(np.float32)
        spectrum = np.fft.fftshift(np.fft.fft2((small - small.mean()) * window))
        power = np.abs(spectrum) ** 2
        cy, cx = sh / 2.0, sw / 2.0
        y, x = np.ogrid[:sh, :sw]
        radius = np.sqrt(((x - cx) / sw) ** 2 + ((y - cy) / sh) ** 2)

        band = (radius > 0.03) & (radius < 0.50)
        log_r = np.log10(radius[band] + 1e-6)
        log_p = np.log10(power[band] + 1e-12)
        slope = float(np.polyfit(log_r, log_p, 1)[0]) if log_r.size > 100 else SLOPE_CENTER
        hf_ratio = float(power[radius >= 0.5].sum() / (power[band].sum() + 1e-9))

        high_pass = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
        local_var = cv2.boxFilter(high_pass * high_pass, -1, (16, 16), normalize=True)
        p50, p95 = np.percentile(local_var, 50), np.percentile(local_var, 95)
        heterogeneity = float(np.log10((p95 + 1e-10) / (p50 + 1e-10)))

        g8 = (gray * 255.0).clip(0, 255).astype(np.uint8)
        residual = gray - cv2.medianBlur(g8, 3).astype(np.float32) / 255.0
        residual_std = float(residual.std())

        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        flat_fraction = float((np.abs(lap) < 0.012).mean())

        green = rgb[..., 1]
        if green.shape[0] >= 4 and green.shape[1] >= 4:
            green_hp = green - cv2.GaussianBlur(green, (0, 0), 1.0)
            parity = np.array(
                [
                    green_hp[0::2, 0::2].mean(),
                    green_hp[0::2, 1::2].mean(),
                    green_hp[1::2, 0::2].mean(),
                    green_hp[1::2, 1::2].mean(),
                ]
            )
            pixel_grid = float(parity.std() / (green_hp.std() + 1e-9))
        else:
            pixel_grid = 0.0

        small_hp = small - cv2.GaussianBlur(small, (0, 0), 1.2)
        small_lap = cv2.Laplacian(small, cv2.CV_32F, ksize=3)
        block = 12
        structure: list[float] = []
        grain_blocks: list[float] = []
        for y0 in range(0, sh - block + 1, block):
            for x0 in range(0, sw - block + 1, block):
                patch = small_hp[y0 : y0 + block, x0 : x0 + block]
                structure.append(float(np.abs(small_lap[y0 : y0 + block, x0 : x0 + block]).mean()))
                grain_blocks.append(float(np.median(np.abs(patch - np.median(patch)))))
        if grain_blocks:
            order = np.argsort(np.asarray(structure))
            calm = order[: max(10, len(order) // 2)]
            smooth_grain = float(np.median(np.asarray(grain_blocks)[calm]))
        else:
            smooth_grain = 0.0

        rgb255 = rgb * 255.0
        r, g, b = rgb255[..., 0], rgb255[..., 1], rgb255[..., 2]
        rg, yb = r - g, 0.5 * (r + g) - b
        colorfulness = float(
            math.sqrt(float(rg.var()) + float(yb.var()))
            + 0.3 * math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
        )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = float(hsv[..., 1].mean())

        return {
            "spectral_slope": slope,
            "hf_ratio": hf_ratio,
            "residual_std": residual_std,
            "noise_heterogeneity": heterogeneity,
            "flat_fraction": flat_fraction,
            "pixel_grid": pixel_grid,
            "smooth_grain": smooth_grain,
            "colorfulness": colorfulness,
            "saturation": saturation,
        }
