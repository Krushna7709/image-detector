"""Decision helpers shared by the engine."""

from __future__ import annotations

import math

from aiforensics.schemas import VerdictLabel


def probability_to_confidence(probability: float) -> float:
    """1 - normalised binary entropy of the AI probability."""

    p = min(max(probability, 1e-6), 1 - 1e-6)
    entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return round(1.0 - entropy, 4)


def decide_label(probability: float, low: float, high: float) -> VerdictLabel:
    if probability >= high:
        return "ai_generated"
    if probability <= low:
        return "real"
    return "uncertain"


def fuse_probabilities(
    model_probability: float | None,
    heuristic_probability: float,
    model_weight: float,
) -> float:
    """Blend a trained detector with classical evidence."""

    if model_probability is None:
        return heuristic_probability
    blended = model_weight * model_probability + (1.0 - model_weight) * heuristic_probability
    return round(min(1.0, max(0.0, blended)), 4)
