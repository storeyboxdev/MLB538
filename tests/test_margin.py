"""Tests for the run-margin regression model."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from mlb_forecaster.config import load_config
from mlb_forecaster.elo.engine import run_engine
from mlb_forecaster.elo.params import EloParams
from mlb_forecaster.ml.features import (MARGIN_LABEL_COL, build_features,
                                        feature_columns)
from mlb_forecaster.ml.margin import MarginModel, fit_margin_model


class _ConstReg:
    """A trivial regressor that always predicts a fixed margin."""

    def __init__(self, mu):
        self.mu = mu

    def predict(self, X):
        return np.full(len(X), self.mu)


def test_margin_label_excluded_from_features():
    cfg = load_config()
    games = pd.DataFrame([
        {"game_pk": 1, "date": "2023-04-01", "season": 2023, "game_type": "R",
         "home_team": "A", "away_team": "B", "home_score": 5, "away_score": 2,
         "status": "Final", "playoff": False, "home_pitcher_id": None,
         "away_pitcher_id": None, "home_starter_id": None, "away_starter_id": None},
        {"game_pk": 2, "date": "2023-04-03", "season": 2023, "game_type": "R",
         "home_team": "B", "away_team": "A", "home_score": 4, "away_score": 1,
         "status": "Final", "playoff": False, "home_pitcher_id": None,
         "away_pitcher_id": None, "home_starter_id": None, "away_starter_id": None},
    ])
    eng = run_engine(games, EloParams())
    feats = build_features(eng, games, cfg)
    assert MARGIN_LABEL_COL in feats.columns
    assert MARGIN_LABEL_COL not in feature_columns(feats)
    # The label equals home_score - away_score.
    assert feats.loc[feats.game_pk == 1, MARGIN_LABEL_COL].iloc[0] == 3


def test_win_prob_matches_normal_cdf():
    model = MarginModel(_ConstReg(2.0), feature_columns=["x"], sigma=4.0)
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    p = model.predict_proba_home(X)
    assert np.allclose(p, norm.cdf(2.0 / 4.0))
    # Positive expected margin -> home favored.
    assert (p > 0.5).all()


def test_sample_margin_centers_on_prediction():
    model = MarginModel(_ConstReg(1.5), feature_columns=["x"], sigma=3.0)
    X = pd.DataFrame({"x": np.zeros(20000)})
    rng = np.random.default_rng(0)
    samples = model.sample_margin(X, rng)
    assert abs(samples.mean() - 1.5) < 0.1
    assert abs(samples.std() - 3.0) < 0.1


def test_fit_margin_model_recovers_signal():
    # Margin driven linearly by one feature; the fitted model should track it.
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(0, 50, n)
    margin = 0.02 * x + rng.normal(0, 3, n)
    feats = pd.DataFrame({"elo_diff": x, MARGIN_LABEL_COL: margin})
    model = fit_margin_model("ridge", load_config(), feats, ["elo_diff"])
    # Higher elo_diff -> higher predicted margin -> higher win prob.
    lo = model.predict_proba_home(pd.DataFrame({"elo_diff": [-100.0]}))[0]
    hi = model.predict_proba_home(pd.DataFrame({"elo_diff": [100.0]}))[0]
    assert hi > 0.5 > lo
    assert 2.0 < model.sigma < 4.0
