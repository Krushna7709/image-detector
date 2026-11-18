from __future__ import annotations

import struct
import zlib


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


def test_health_and_examples(app_client):
    health = app_client.get("/api/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["detector"] in {"heuristic", "clip-linear", "cnn"}
    assert payload["localizer"] in {"heuristic", "unet"}

    examples = app_client.get("/api/examples")
    assert examples.status_code == 200
    entries = examples.json()
    assert len(entries) == 6
    assert {entry["kind"] for entry in entries} == {"ai", "real"}
    assert app_client.get("/examples/ai_mountain.png").status_code == 200


def test_analyze_upload(app_client, image_factory):
    response = app_client.post(
        "/api/analyze",
        files={"file": ("sample.png", image_factory(), "image/png")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["filename"] == "sample.png"
    assert result["verdict"]["label"] in {"real", "ai_generated", "uncertain"}
    assert 0.0 <= result["verdict"]["ai_probability"] <= 1.0
    assert result["metadata"]["format"] == "PNG"
    assert result["edit"]["method"] in {"heuristic", "unet"}
    assert result["edit"]["heatmap"].startswith("data:image/png;base64,")
    assert result["explain"]["preview"].startswith("data:image/png;base64,")
    assert len(result["signals"]) == 9
    assert result["notes"]
    assert "decision support" in " ".join(result["notes"])


def test_analyze_with_distortion(app_client, image_factory):
    response = app_client.post(
        "/api/analyze",
        files={"file": ("sample.png", image_factory(), "image/png")},
        data={"distortion": "jpeg30"},
    )
    assert response.status_code == 200
    assert response.json()["distortion"] == "JPEG q30"


def test_analyze_example(app_client):
    response = app_client.post("/api/analyze/example/ai_mountain")
    assert response.status_code == 200
    result = response.json()
    assert result["metadata"]["has_c2pa"] is True
    assert result["metadata"]["generator_hint"] == "OpenAI ImageGen"
    assert any(item["source"] == "metadata" for item in result["attribution"])


def test_unknown_example_is_404(app_client):
    assert app_client.post("/api/analyze/example/does-not-exist").status_code == 404


def test_invalid_upload_and_distortion(app_client):
    response = app_client.post(
        "/api/analyze",
        files={"file": ("broken.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_format"

    response = app_client.post(
        "/api/analyze",
        files={"file": ("sample.png", b"unused", "image/png")},
        data={"distortion": "sepia"},
    )
    assert response.status_code == 400


def test_decompression_bomb_is_rejected(app_client):
    response = app_client.post(
        "/api/analyze",
        files={"file": ("bomb.png", _png_header(100_000, 100_000), "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "too_many_pixels"

    response = app_client.post(
        "/api/analyze",
        files={"file": ("bomb.png", _png_header(100_000, 100_000), "image/png")},
        data={"distortion": "jpeg50"},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "too_many_pixels"


def test_batch_analysis(app_client, image_factory):
    files = [
        ("files", ("one.png", image_factory(seed=1), "image/png")),
        ("files", ("two.png", image_factory(seed=2), "image/png")),
    ]
    response = app_client.post("/api/analyze/batch", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["errors"] == []
    assert {result["filename"] for result in payload["results"]} == {"one.png", "two.png"}


def test_security_headers(app_client):
    response = app_client.get("/api/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"
