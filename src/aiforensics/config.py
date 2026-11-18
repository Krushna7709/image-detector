"""Central configuration.

Paths can be overridden with environment variables so the service can run
inside containers or from a different working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name)
    return Path(raw).expanduser().resolve() if raw else default


def _model_file(env_name: str, filename: str) -> Path:
    raw = os.getenv(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return _path("FORENSIQ_MODEL_DIR", PROJECT_ROOT / "models") / filename


@dataclass(frozen=True)
class Settings:
    app_name: str = "ForensIQ"
    version: str = "0.1.0"
    description: str = (
        "Detect, localize and attribute AI-generated images with calibrated, "
        "explainable forensic evidence."
    )

    project_root: Path = PROJECT_ROOT
    model_dir: Path = field(
        default_factory=lambda: _path("FORENSIQ_MODEL_DIR", PROJECT_ROOT / "models")
    )
    detector_path: Path = field(
        default_factory=lambda: _model_file("FORENSIQ_DETECTOR_PATH", "detector.joblib")
    )
    attribution_path: Path = field(
        default_factory=lambda: _model_file("FORENSIQ_ATTRIBUTION_PATH", "attribution.joblib")
    )
    cnn_path: Path = field(
        default_factory=lambda: _model_file("FORENSIQ_CNN_PATH", "cnn_detector.pt")
    )
    localizer_path: Path = field(
        default_factory=lambda: _model_file("FORENSIQ_LOCALIZER_PATH", "localizer.pt")
    )

    clip_model_name: str = os.getenv("FORENSIQ_CLIP_MODEL", "openai/clip-vit-base-patch32")
    clip_device: str = os.getenv("FORENSIQ_CLIP_DEVICE", "auto")

    max_upload_bytes: int = 15 * 1024 * 1024
    max_pixels: int = 50_000_000
    analysis_max_side: int = 1024
    display_max_side: int = 768
    thumbnail_max_side: int = 384
    cache_size: int = 32

    allowed_formats: tuple[str, ...] = ("JPEG", "PNG", "WEBP", "BMP", "TIFF")
    decision_threshold: float = 0.55
    uncertain_low: float = 0.42
    uncertain_high: float = 0.62

    # Weight of the trained detector relative to the classical evidence score
    # when both are available. The trained head dominates but the heuristic
    # still nudges genuinely ambiguous cases.
    model_fusion_weight: float = 0.75

    distortions: tuple[str, ...] = ("none", "jpeg50", "jpeg30", "resize50", "blur2")

    def capability_summary(self) -> dict[str, object]:
        return {
            "detector": self._detector_mode(),
            "attribution": self.attribution_path.exists(),
            "localizer": "unet" if self.localizer_path.exists() else "heuristic",
            "clip_backbone": self.clip_model_name,
            "clip_loaded": False,
        }

    def _detector_mode(self) -> str:
        if self.cnn_path.exists():
            return "cnn"
        if self.detector_path.exists():
            return "clip-linear"
        return "heuristic"


settings = Settings()
