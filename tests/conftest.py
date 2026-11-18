from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def image_factory():
    """Create deterministic photo-like RGB images for tests."""

    def _make(
        width: int = 320,
        height: int = 240,
        seed: int = 0,
        fmt: str = "PNG",
        **save_kwargs,
    ) -> bytes:
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        base = (
            0.5
            + 0.16 * np.sin(xx / 17.0)
            + 0.11 * np.cos(yy / 23.0)
            + rng.normal(0.0, 0.035, (height, width))
        )
        channels = np.stack(
            [
                base * 1.02 + 0.03,
                base * 0.96 + 0.04,
                base * 0.88,
            ],
            axis=-1,
        )
        array = np.clip(channels, 0, 1)
        array = (array * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array, "RGB").save(buffer, format=fmt, **save_kwargs)
        return buffer.getvalue()

    return _make


@pytest.fixture(scope="session")
def app_client():
    from fastapi.testclient import TestClient

    from aiforensics.api.app import app

    with TestClient(app) as client:
        yield client
