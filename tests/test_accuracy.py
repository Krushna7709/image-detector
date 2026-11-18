"""Regression checks for the bundled demo set.

These keep the classical engine from confidently flipping AI samples to "real"
after recompression or confidently flagging blurred photographs as synthetic.
"""

from __future__ import annotations

from pathlib import Path

from aiforensics.config import Settings
from aiforensics.core.engine import ForensicsEngine

EXAMPLES = Path(__file__).resolve().parents[1] / "src" / "aiforensics" / "api" / "examples"
AI_SAMPLES = ("ai_mountain.png", "ai_portrait.png", "ai_product.png")
REAL_SAMPLES = ("real_cat_portrait.jpg", "real_mountain_lake.jpg", "real_street_market.jpg")


def _engine(tmp_path) -> ForensicsEngine:
    return ForensicsEngine(
        Settings(
            model_dir=tmp_path,
            detector_path=tmp_path / "detector.joblib",
            attribution_path=tmp_path / "attribution.joblib",
            cnn_path=tmp_path / "cnn_detector.pt",
            localizer_path=tmp_path / "localizer.pt",
        )
    )


def test_bundled_examples_hold_up_to_distortion(tmp_path):
    engine = _engine(tmp_path)

    for name in AI_SAMPLES:
        data = (EXAMPLES / name).read_bytes()
        assert engine.analyze(data, name).verdict.label == "ai_generated"
        for distortion in ("jpeg50", "jpeg30", "resize50"):
            result = engine.analyze(data, name, distortion=distortion)
            assert result.verdict.label != "real", (
                f"{name} ({distortion}) flipped to real at p={result.verdict.ai_probability}"
            )

    for name in REAL_SAMPLES:
        data = (EXAMPLES / name).read_bytes()
        assert engine.analyze(data, name).verdict.label == "real"
        for distortion in ("jpeg50", "jpeg30", "blur2"):
            result = engine.analyze(data, name, distortion=distortion)
            assert result.verdict.label != "ai_generated", (
                f"{name} ({distortion}) flipped to synthetic at p={result.verdict.ai_probability}"
            )
