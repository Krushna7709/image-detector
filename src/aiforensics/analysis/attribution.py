"""Open-set generator attribution head over CLIP embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttributionModel:
    pipeline: object
    classes: list[str]
    threshold: float = 0.40
    trained_on: list[str] = field(default_factory=list)

    @classmethod
    def load_if_available(cls, path: Path) -> "AttributionModel | None":
        if not path.exists():
            return None
        try:
            import joblib

            artifact = joblib.load(path)
            if not isinstance(artifact, dict):
                raise ValueError("unexpected artifact format")
            pipeline = artifact["pipeline"]
            classes = [str(c) for c in artifact["classes"]]
            return cls(
                pipeline=pipeline,
                classes=classes,
                threshold=float(artifact.get("threshold", 0.40)),
                trained_on=list(artifact.get("trained_on", classes)),
            )
        except Exception as exc:
            logger.warning("Could not load attribution head from %s: %s", path, exc)
            return None

    def predict(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        probabilities = self.pipeline.predict_proba(np.asarray(embedding).reshape(1, -1))[0]
        pairs = sorted(zip(self.classes, probabilities), key=lambda item: item[1], reverse=True)
        return [(name, round(float(prob), 4)) for name, prob in pairs]
