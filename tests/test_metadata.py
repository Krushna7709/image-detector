from __future__ import annotations

import io

from PIL import Image, PngImagePlugin

from aiforensics.config import settings
from aiforensics.metadata import extract_metadata
from aiforensics.preprocessing import load_image


def _with_png_text(data: bytes, key: str, value: str) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        info = PngImagePlugin.PngInfo()
        info.add_text(key, value)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def test_generator_hint_from_png_text(image_factory):
    data = _with_png_text(image_factory(), "Software", "Midjourney v6.1")
    bundle = extract_metadata(data, load_image(data, "sample.png", settings))
    assert bundle.report.generator_hint == "Midjourney"
    assert "Software" in bundle.report.png_text_keys
    assert bundle.report.has_c2pa is False


def test_c2pa_marker_and_synthetic_source_type(image_factory):
    data = image_factory() + b"\x00c2pa trainedAlgorithmicMedia"
    bundle = extract_metadata(data, load_image(data, "sample.png", settings))
    assert bundle.report.has_c2pa is True
    assert bundle.report.synthetic_provenance is True
    assert bundle.report.c2pa_source_type == "trainedAlgorithmicMedia"


def test_watermark_marker_is_reported(image_factory):
    data = image_factory() + b"SynthID"
    bundle = extract_metadata(data, load_image(data, "sample.png", settings))
    assert bundle.report.watermark_marker == "synthid"


def test_camera_exif_and_jpeg_quality(image_factory):
    with Image.open(io.BytesIO(image_factory())) as image:
        exif = Image.Exif()
        exif[271] = "Canon"  # Make
        exif[272] = "EOS R6"  # Model
        exif[305] = "Lightroom 13"  # Software
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=70, exif=exif)
    data = buffer.getvalue()
    bundle = extract_metadata(data, load_image(data, "camera.jpg", settings))
    assert bundle.report.camera == "Canon EOS R6"
    assert bundle.report.software == "Lightroom 13"
    assert bundle.report.has_exif is True
    assert bundle.report.generator_hint is None
    assert bundle.report.jpeg_quality_estimate is not None
    assert 60 <= bundle.report.jpeg_quality_estimate <= 80
