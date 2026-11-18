from __future__ import annotations

import io

from PIL import Image, PngImagePlugin

from aiforensics.config import Settings
from aiforensics.core.engine import ForensicsEngine


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


def test_engine_runs_heuristic_pipeline(tmp_path, image_factory):
    engine = _engine(tmp_path)
    assert engine.health().detector == "heuristic"
    result = engine.analyze(image_factory(), "photo.png")
    assert result.engine == "heuristic"
    assert result.verdict.label in {"real", "uncertain", "ai_generated"}
    assert result.edit.method == "heuristic"
    assert result.explain.gradcam is None
    assert result.metadata.format == "PNG"
    assert result.filename == "photo.png"
    assert len(result.signals) == 7
    assert any("decision support" in note for note in result.notes)


def test_engine_caches_by_content(tmp_path, image_factory):
    engine = _engine(tmp_path)
    data = image_factory(seed=3)
    first = engine.analyze(data, "first.png")
    second = engine.analyze(data, "second.png")
    assert second.sha256 == first.sha256
    assert first.filename == "first.png"
    assert second.filename == "second.png"
    assert second is not first
    assert len(engine._cache) == 1


def test_engine_treats_degenerate_images_as_inconclusive(tmp_path):
    engine = _engine(tmp_path)
    for size in (1, 8):
        buffer = io.BytesIO()
        Image.new("RGB", (size, size), (120, 130, 140)).save(buffer, format="PNG")
        result = engine.analyze(buffer.getvalue(), f"flat-{size}.png")
        assert result.verdict.label == "uncertain"
        assert result.verdict.ai_probability == 0.5
        assert result.verdict.reliability <= 0.05
        assert any("too small or too uniform" in note for note in result.notes)


def test_engine_records_distortion(tmp_path, image_factory):
    engine = _engine(tmp_path)
    result = engine.analyze(image_factory(), "photo.png", distortion="jpeg30")
    assert result.distortion == "JPEG q30"
    assert any("simulated copy" in note for note in result.notes)


def test_engine_sanitises_filename(tmp_path, image_factory):
    engine = _engine(tmp_path)
    result = engine.analyze(image_factory(), "../../etc/passwd.png")
    assert result.filename == "passwd.png"


def test_metadata_hint_forces_ai_verdict(tmp_path, image_factory):
    with Image.open(io.BytesIO(image_factory())) as image:
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", "Midjourney v6.1")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", pnginfo=info)
    engine = _engine(tmp_path)
    result = engine.analyze(buffer.getvalue(), "render.png")
    assert result.verdict.label == "ai_generated"
    assert result.verdict.ai_probability >= 0.90
    assert result.attribution[0].generator == "Midjourney"
    assert result.attribution[0].source == "metadata"
