from __future__ import annotations

import numpy as np
import pytest

from aiforensics.analysis.heuristic import HeuristicAnalyzer
from aiforensics.analysis.localization import (
    HeuristicLocalizer,
    regions_from_heat,
    summarize_heat,
)
from aiforensics.config import settings
from aiforensics.core.fusion import (
    decide_label,
    fuse_probabilities,
    probability_to_confidence,
)
from aiforensics.preprocessing import load_image

from aiforensics.schemas import EditRegion


def _rgb(image_factory) -> np.ndarray:
    return load_image(image_factory(), "sample.png", settings).analysis_rgb


def test_heuristic_signals_and_ranges(image_factory):
    result = HeuristicAnalyzer().analyze(
        _rgb(image_factory),
        generator_hint=None,
        has_camera_exif=False,
        jpeg_quality=None,
    )
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.reliability <= 1.0
    assert len(result.signals) == 9
    assert sum(signal.weight for signal in result.signals) == pytest.approx(1.0)
    assert all(0.0 <= signal.score <= 1.0 for signal in result.signals)


def test_provenance_floor(image_factory):
    rgb = _rgb(image_factory)
    result = HeuristicAnalyzer().analyze(
        rgb, generator_hint="Midjourney", has_camera_exif=False, jpeg_quality=None
    )
    assert result.probability >= 0.90
    provenance = next(signal for signal in result.signals if signal.name == "provenance")
    assert provenance.score == 0.97

    synthetic = HeuristicAnalyzer().analyze(
        rgb, generator_hint=None, has_camera_exif=False, jpeg_quality=None, synthetic_provenance=True
    )
    assert synthetic.probability >= 0.90


def test_camera_exif_lowers_provenance_score(image_factory):
    result = HeuristicAnalyzer().analyze(
        _rgb(image_factory),
        generator_hint=None,
        has_camera_exif=True,
        jpeg_quality=90,
    )
    provenance = next(signal for signal in result.signals if signal.name == "provenance")
    assert provenance.score == 0.12
    assert provenance.reliability == 0.55


def test_fusion_and_verdict_bands():
    assert decide_label(0.10, 0.42, 0.62) == "real"
    assert decide_label(0.42, 0.42, 0.62) == "real"
    assert decide_label(0.50, 0.42, 0.62) == "uncertain"
    assert decide_label(0.619, 0.42, 0.62) == "uncertain"
    assert decide_label(0.62, 0.42, 0.62) == "ai_generated"

    assert fuse_probabilities(None, 0.3, 0.75) == 0.3
    assert fuse_probabilities(0.8, 0.4, 0.75) == pytest.approx(0.7)
    assert probability_to_confidence(0.5) == 0.0
    assert 0.65 < probability_to_confidence(0.95) < 0.80


def test_heat_regions_are_extracted():
    heat = np.zeros((160, 160), dtype=np.float32)
    heat[55:105, 55:105] = 1.0
    top_mean, area_ratio, probability = summarize_heat(heat)
    assert top_mean > 0.9
    assert area_ratio > 0.09
    assert probability > 0.5

    regions = regions_from_heat(heat)
    assert regions
    assert any(_contains(region, 80, 80) for region in regions)
    assert all(region.score > 0.8 for region in regions)


def _contains(region: EditRegion, x: int, y: int) -> bool:
    return region.x <= x <= region.x + region.width and region.y <= y <= region.y + region.height


def test_flat_heat_is_not_an_edit():
    heat = np.zeros((128, 128), dtype=np.float32)
    assert summarize_heat(heat) == (0.0, 0.0, 0.0)
    assert regions_from_heat(heat) == []


def test_uniform_noise_does_not_trigger_localizer():
    rng = np.random.default_rng(4)
    rgb = rng.integers(70, 190, (128, 128, 3), dtype=np.uint8)
    result = HeuristicLocalizer().localize(rgb)
    assert result.method == "heuristic"
    assert result.edited_probability <= 0.2
    assert result.regions == []
    assert result.heat.shape == rgb.shape[:2]
