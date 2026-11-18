"""Reproducible dataset, training and evaluation pipelines.

Modules are intentionally not imported here so the core service keeps working
without torch/scikit-learn; run them with ``python -m aiforensics.training.<name>``.
"""

from __future__ import annotations

__all__ = [
    "manifest",
    "augmentations",
    "datasets",
    "metrics",
    "calibration",
    "clip_features",
    "train_detector",
    "train_attribution",
    "train_cnn",
    "train_localizer",
    "evaluate",
    "generate",
    "make_edits",
]
