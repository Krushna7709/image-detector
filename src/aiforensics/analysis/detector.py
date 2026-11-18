"""Linear detector head over CLIP embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LinearDetector:
    pipeline: object
    threshold: float = 0.5
    trained_on: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    calibrated: bool = False

    @classmethod
    def load_if_available(cls, path: Path) -> "LinearDetector | None":
        if not path.exists():
            return None
        try:
            import joblib

            artifact = joblib.load(path)
            if isinstance(artifact, dict):
                pipeline = artifact.get("pipeline") or artifact.get("model")
            else:
                pipeline, artifact = artifact, {}
            if pipeline is None or not hasattr(pipeline, "predict_proba"):
                raise ValueError("artifact does not contain a classifier with predict_proba")
            return cls(
                pipeline=pipeline,
                threshold=float(artifact.get("threshold", 0.5)),
                trained_on=list(artifact.get("trained_on", [])),
                metrics=dict(artifact.get("metrics", {})),
                calibrated=bool(artifact.get("calibrated", False)),
            )
        except Exception as exc:
            logger.warning("Could not load detector head from %s: %s", path, exc)
            return None

    def predict(self, embedding: np.ndarray) -> float:
        probabilities = self.pipeline.predict_proba(np.asarray(embedding).reshape(1, -1))[0]
        classes = list(getattr(self.pipeline, "classes_", [0, 1]))
        if 1 in classes:
            return float(probabilities[classes.index(1)])
        return float(probabilities[-1])
