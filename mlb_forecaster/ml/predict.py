"""Load a saved model and produce calibrated home-win probabilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .registry import load_model


class Predictor:
    """Thin wrapper over a saved CalibratedModel for forecasting."""

    def __init__(self, calibrated_model, cols: list[str]):
        self.model = calibrated_model
        self.cols = cols

    @classmethod
    def from_registry(cls, models_dir: Path, version: str = "latest") -> "Predictor":
        bundle = load_model(models_dir, version)
        return cls(bundle["model"], bundle["cols"])

    def predict_home_prob(self, feats: pd.DataFrame):
        """Return calibrated P(home win) for each row of a feature matrix."""
        missing = [c for c in self.cols if c not in feats.columns]
        if missing:
            raise ValueError(f"Feature matrix missing columns: {missing}")
        return self.model.predict_proba_home(feats)
