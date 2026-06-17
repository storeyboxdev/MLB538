"""Walk-forward backtest comparing Elo baselines vs the ML models.

Produces, on a common pooled out-of-sample test set (seasons after the minimum
training window), log loss / Brier for: the team-only Elo probability, the
pitcher-adjusted Elo probability, logistic regression, and LightGBM — plus a
calibration table for the best model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..config import Config
from ..elo.engine import run_engine
from ..elo.params import EloParams
from ..ml.features import build_features, feature_columns
from ..ml.margin import walk_forward_cv as margin_cv
from ..ml.train import walk_forward_cv

_EPS = 1e-12


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Decile calibration: predicted vs observed home-win rate per bucket."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        out.append({
            "bucket": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(mask.sum()),
            "pred_mean": round(float(p[mask].mean()), 4),
            "obs_rate": round(float(y[mask].mean()), 4),
        })
    return out


def _metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(p, _EPS, 1 - _EPS)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "n": int(len(y))}


def run_backtest(config: Config, games: pd.DataFrame,
                 params: EloParams) -> dict[str, Any]:
    """Run the full backtest and return a JSON-serializable report."""
    eng = run_engine(games, params)
    feats = build_features(eng, games, config)
    cols = feature_columns(feats)
    method = config.raw["ml"].get("calibration", "isotonic")
    min_train = config.raw["ml"]["cv"].get("min_train_seasons", 2)

    seasons = sorted(feats["season"].unique())
    test_seasons = seasons[min_train:]

    # Elo baselines on the same pooled test seasons' final games.
    elo_df = eng[(eng["home_win"].notna()) & (eng["season"].isin(test_seasons))]
    y_elo = elo_df["home_win"].to_numpy()
    elo_team = _metrics(elo_df["elo_prob1"].to_numpy(), y_elo)
    elo_rating = _metrics(elo_df["rating_prob1"].to_numpy(), y_elo)

    # ML models via walk-forward CV (same test seasons).
    lr = walk_forward_cv("logistic", config, feats, cols, method)
    gbm = walk_forward_cv("lightgbm", config, feats, cols, method)
    # Run-margin model: derived win-prob metrics + margin error.
    mreg = margin_cv("ridge", config, feats, cols)

    models = {
        "elo_team_only": elo_team,
        "elo_pitcher_adj": elo_rating,
        "logistic": {k: lr[k] for k in ("log_loss", "brier", "n")},
        "lightgbm": {k: gbm[k] for k in ("log_loss", "brier", "n")},
        "margin_ridge": {k: mreg[k] for k in
                         ("log_loss", "brier", "n", "margin_mae", "margin_rmse")},
    }
    best_name = min(models, key=lambda k: models[k]["log_loss"])

    pooled = {"logistic": lr, "lightgbm": gbm, "margin_ridge": mreg}
    # Calibration table for the best model (fall back to Elo if an Elo wins).
    if best_name in pooled:
        calib = calibration_table(pooled[best_name]["_p"], pooled[best_name]["_y"])
    else:
        col = "rating_prob1" if best_name == "elo_pitcher_adj" else "elo_prob1"
        calib = calibration_table(elo_df[col].to_numpy(), y_elo)

    return {
        "test_seasons": [int(s) for s in test_seasons],
        "models": models,
        "best_model": best_name,
        "calibration_best": calib,
        "ml_folds": {"logistic": lr["folds"], "lightgbm": gbm["folds"]},
    }
