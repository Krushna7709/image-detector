"""Probability calibration helpers.

The linear head is wrapped in a Platt sigmoid fitted on validation data, which
is pickled together with the pipeline so the runtime gets calibrated scores.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


class PlattCalibrator:
    """One-dimensional logistic calibration of a probability score."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    def fit(self, probabilities, y_true) -> "PlattCalibrator":
        from scipy.optimize import minimize

        p = np.asarray(probabilities, dtype=np.float64).ravel()
        y = np.asarray(y_true, dtype=np.float64).ravel()
        if p.size == 0 or len(np.unique(y)) < 2:
            raise ValueError("Calibration needs samples from both classes.")
        z = _logit(p)

        def objective(params: np.ndarray) -> float:
            a, b = float(params[0]), float(params[1])
            q = np.clip(_expit(a * z + b), _EPS, 1.0 - _EPS)
            nll = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)).mean()
            return float(nll + 1e-6 * (a - 1.0) ** 2)

        result = minimize(objective, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
        if result.success:
            self.a, self.b = float(result.x[0]), float(result.x[1])
        return self

    def predict(self, probabilities) -> np.ndarray:
        z = _logit(np.asarray(probabilities, dtype=np.float64))
        return _expit(self.a * z + self.b)

    def to_dict(self) -> dict[str, float]:
        return {"type": "platt", "a": self.a, "b": self.b}


class CalibratedClassifier:
    """``predict_proba`` wrapper matching scikit-learn's classifier contract."""

    def __init__(self, base, calibrator: PlattCalibrator):
        self.base = base
        self.calibrator = calibrator
        self.classes_ = np.asarray(base.classes_)

    def predict_proba(self, features) -> np.ndarray:
        raw = np.asarray(self.base.predict_proba(features), dtype=np.float64)
        positive_column = int(np.where(self.classes_ == 1)[0][0])
        calibrated = self.calibrator.predict(raw[:, positive_column])
        output = np.zeros_like(raw)
        for column, label in enumerate(self.classes_):
            output[:, column] = calibrated if label == 1 else 1.0 - calibrated
        return output


def fit_calibrator(probabilities, y_true) -> PlattCalibrator | None:
    try:
        return PlattCalibrator().fit(probabilities, y_true)
    except (ValueError, ImportError):  # single-class or missing scipy
        return None
