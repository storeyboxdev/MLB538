"""Probability calibration wrappers (isotonic / Platt sigmoid).

We keep calibration explicit and version-stable rather than relying on
sklearn's evolving prefit API: a base estimator is fit on one slice, and a 1-D
calibrator maps its raw probabilities to calibrated ones using a held-out slice.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratedModel:
    """Wraps a fitted base classifier + optional 1-D probability calibrator."""

    def __init__(self, base, feature_columns: list[str], method: str = "isotonic",
                 calibrator=None):
        self.base = base
        self.feature_columns = feature_columns
        self.method = method
        self.calibrator = calibrator

    def _raw(self, X) -> np.ndarray:
        X = X[self.feature_columns]
        return self.base.predict_proba(X)[:, 1]

    def predict_proba_home(self, X) -> np.ndarray:
        """Calibrated probability the home team wins."""
        p = self._raw(X)
        if self.calibrator is None:
            return p
        if self.method == "isotonic":
            return self.calibrator.predict(p)
        # sigmoid / Platt
        return self.calibrator.predict_proba(p.reshape(-1, 1))[:, 1]


def fit_calibrator(method: str, raw_probs: np.ndarray, y: np.ndarray):
    """Fit a 1-D calibrator mapping raw probabilities -> calibrated probabilities."""
    if method in (None, "none"):
        return None
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(raw_probs, y)
        return cal
    if method == "sigmoid":
        cal = LogisticRegression()
        cal.fit(raw_probs.reshape(-1, 1), y)
        return cal
    raise ValueError(f"Unknown calibration method: {method}")
