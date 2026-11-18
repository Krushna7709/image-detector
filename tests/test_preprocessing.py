from __future__ import annotations

import io
import struct
import zlib
from dataclasses import replace

import pytest
from PIL import Image

from aiforensics.config import settings
from aiforensics.preprocessing import ImageValidationError, distort_image, load_image


def _png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def test_load_image_reports_core_properties(image_factory):
    loaded = load_image(image_factory(width=640, height=480), "sample.png", settings)
    assert loaded.format == "PNG"
    assert (loaded.width, loaded.height) == (640, 480)
    assert len(loaded.sha256) == 64
    assert loaded.analysis_rgb.shape[:2] == loaded.display_rgb.shape[:2]
    assert max(loaded.analysis_rgb.shape[:2]) <= settings.analysis_max_side
    assert loaded.analysis.dtype.name == "float32"


def test_empty_and_corrupt_files(image_factory):
    with pytest.raises(ImageValidationError) as empty:
        load_image(b"", "empty.png", settings)
    assert empty.value.code == "empty_file"

    with pytest.raises(ImageValidationError) as corrupt:
        load_image(b"not an image", "corrupt.bin", settings)
    assert corrupt.value.code == "unsupported_format"


def test_size_and_pixel_limits(image_factory):
    tiny_limit = replace(settings, max_upload_bytes=8)
    with pytest.raises(ImageValidationError) as too_large:
        load_image(image_factory(), "sample.png", tiny_limit)
    assert too_large.value.code == "file_too_large"
    assert too_large.value.status_code == 413

    pixel_limit = replace(settings, max_pixels=100)
    with pytest.raises(ImageValidationError) as too_many:
        load_image(image_factory(width=64, height=64), "sample.png", pixel_limit)
    assert too_many.value.code == "too_many_pixels"


def test_disallowed_format():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 140, 160)).save(buffer, format="GIF")
    with pytest.raises(ImageValidationError) as exc:
        load_image(buffer.getvalue(), "sample.gif", settings)
    assert exc.value.code == "unsupported_format"
    assert exc.value.status_code == 415


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("jpeg50", "JPEG q50"),
        ("jpeg30", "JPEG q30"),
        ("resize50", "Resized 50% and restored"),
        ("blur2", "Gaussian blur (sigma 2)"),
    ],
)
def test_distortions_round_trip(name, label, image_factory):
    data, returned_label = distort_image(image_factory(width=200, height=160), name)
    assert returned_label == label
    loaded = load_image(data, "distorted.png", settings)
    assert (loaded.width, loaded.height) == (200, 160)


def test_unknown_distortion_is_rejected(image_factory):
    with pytest.raises(ImageValidationError):
        distort_image(image_factory(), "sepia")


def test_oversized_images_are_rejected_before_decoding():
    data = _png_header(8000, 8000)
    with pytest.raises(ImageValidationError) as exc:
        load_image(data, "huge.png", settings)
    assert exc.value.code == "too_many_pixels"
    assert exc.value.status_code == 413

    with pytest.raises(ImageValidationError) as distorted:
        distort_image(data, "jpeg50", settings)
    assert distorted.value.code == "too_many_pixels"


def test_decompression_bomb_raises_validation_error():
    data = _png_header(100_000, 100_000)
    with pytest.raises(ImageValidationError) as exc:
        load_image(data, "bomb.png", settings)
    assert exc.value.code == "too_many_pixels"
    assert exc.value.status_code == 413


def test_exif_orientation_is_applied():
    image = Image.new("RGB", (100, 50), (90, 120, 160))
    exif = Image.Exif()
    exif[274] = 6  # rotate 270 degrees, swapping width and height
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    loaded = load_image(buffer.getvalue(), "rotated.jpg", settings)
    assert (loaded.width, loaded.height) == (50, 100)
