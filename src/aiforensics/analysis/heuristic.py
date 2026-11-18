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
# decision boundaries; the fusion step combines them.
NOISE_HET_PIVOT = 2.4       # log10(p95/p50) of local noise variance
DETAIL_PIVOT = 0.26         # share of near-flat pixels
DETAIL_SPREAD = 0.20
RESIDUAL_PIVOT = 0.028      # median-filter residual standard deviation
RESIDUAL_SPREAD = 0.018
SLOPE_CENTER = -2.6         # radial log-log power slope
SLOPE_SPREAD = 1.3
HF_PIVOT = 0.020            # energy above half-Nyquist
SAT_PIVOT = 0.42
COLOR_PIVOT = 45.0

FUSION_CENTER = 0.32
FUSION_SCALE = 4.2

WEIGHTS = {
    "noise_heterogeneity": 0.30,
    "detail_distribution": 0.16,
    "noise_floor": 0.10,
    "spectral_decay": 0.10,
    "high_frequency": 0.05,
    "color_signature": 0.06,
    "provenance": 0.15,
}

LABELS = {
    "noise_heterogeneity": "Noise field heterogeneity",
    "detail_distribution": "Detail distribution",
    "noise_floor": "Residual noise floor",
    "spectral_decay": "Spectral decay",
    "high_frequency": "High-frequency balance",
    "color_signature": "Color signature",
    "provenance": "Provenance metadata",
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
    ) -> HeuristicResult:
        rgb = rgb.astype(np.float32) / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        features = self._features(gray, rgb)

        compressed = jpeg_quality is not None and jpeg_quality < 60
        freq_rel = 0.72 if compressed else 1.0
        noise_rel = 0.82 if compressed else 1.0

        noise_score = _clip01(1.0 - features["noise_heterogeneity"] / NOISE_HET_PIVOT)
        detail_score = _clip01((DETAIL_PIVOT - features["flat_fraction"]) / DETAIL_SPREAD)
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

        signals = [
            Signal(
                name="noise_heterogeneity",
                label=LABELS["noise_heterogeneity"],
                score=round(noise_score, 4),
                weight=WEIGHTS["noise_heterogeneity"],
                reliability=round(0.62 * noise_rel, 4),
                detail=(
                    f"Local noise variance spread p95/p50 = "
                    f"{10 ** features['noise_heterogeneity']:.1f} "
                    f"(log10 {features['noise_heterogeneity']:.2f}). Uniform fields lean synthetic."
                ),
            ),
            Signal(
                name="detail_distribution",
                label=LABELS["detail_distribution"],
                score=round(detail_score, 4),
                weight=WEIGHTS["detail_distribution"],
                reliability=round(0.40 * freq_rel, 4),
                detail=(
                    f"Near-flat pixel share {features['flat_fraction']:.0%}; "
                    "camera photos usually mix flat and textured areas."
                ),
            ),
            Signal(
                name="noise_floor",
                label=LABELS["noise_floor"],
                score=round(residual_score, 4),
                weight=WEIGHTS["noise_floor"],
                reliability=round(0.50 * noise_rel, 4),
                detail=(
                    f"Median-filter residual std {features['residual_std']:.4f}. "
                    "Very low floors are common in renders and heavy denoising."
                ),
            ),
            Signal(
                name="spectral_decay",
                label=LABELS["spectral_decay"],
                score=round(slope_score, 4),
                weight=WEIGHTS["spectral_decay"],
                reliability=round(0.40 * freq_rel, 4),
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
                reliability=round(0.30 * freq_rel, 4),
                detail=f"Energy above half-Nyquist: {features['hf_ratio']:.3f}. Weakly informative alone.",
            ),
            Signal(
                name="color_signature",
                label=LABELS["color_signature"],
                score=round(color_score, 4),
                weight=WEIGHTS["color_signature"],
                reliability=0.25,
                detail=(
                    f"Colorfulness {features['colorfulness']:.1f}, "
                    f"mean saturation {features['saturation']:.2f}."
                ),
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
        if features["noise_heterogeneity"] < 1.5:
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
            "colorfulness": colorfulness,
            "saturation": saturation,
        }
