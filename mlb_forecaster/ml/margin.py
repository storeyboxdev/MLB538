"""Run-margin regression model.

Predicts the home team's run margin (home_score - away_score) and treats it as
Normal(mu_hat, sigma) with a fitted residual sigma. From that we derive a win
probability P(margin > 0) = Phi(mu_hat / sigma) and can sample realistic run
differentials for the season simulation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config
from .features import LABEL_COL, MARGIN_LABEL_COL, feature_columns

_EPS = 1e-12


class MarginModel:
    """Wraps a fitted margin regressor + residual sigma (homoscedastic)."""

    def __init__(self, reg, feature_columns: list[str], sigma: float):
        self.reg = reg
        self.feature_columns = feature_columns
        self.sigma = max(float(sigma), 1e-3)

    def predict_margin(self, X) -> np.ndarray:
        return self.reg.predict(X[self.feature_columns])

    def predict_proba_home(self, X) -> np.ndarray:
        """Win probability derived from the margin distribution."""
        mu = self.predict_margin(X)
        return norm.cdf(mu / self.sigma)

    def sample_margin(self, X, rng: np.random.Generator) -> np.ndarray:
        """Draw a run margin per game: mu_hat + sigma * N(0, 1)."""
        mu = self.predict_margin(X)
        return mu + self.sigma * rng.standard_normal(len(mu))


def make_regressor(kind: str, config: Config):
    if kind == "ridge":
        alpha = config.raw["ml"].get("margin", {}).get("ridge_alpha", 1.0)
        return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=alpha))])
    if kind == "lightgbm":
        p = config.raw["ml"]["lightgbm"]
        return LGBMRegressor(
            n_estimators=p.get("n_estimators", 400),
            learning_rate=p.get("learning_rate", 0.03),
            num_leaves=p.get("num_leaves", 31),
            max_depth=p.get("max_depth", -1),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            min_child_samples=p.get("min_child_samples", 50),
            reg_lambda=p.get("reg_lambda", 1.0),
            verbosity=-1,
        )
    raise ValueError(f"Unknown margin model kind: {kind}")


def fit_margin_model(kind: str, config: Config, feats: pd.DataFrame,
                     cols: list[str]) -> MarginModel:
    """Fit the regressor and estimate the residual sigma on the training data."""
    X = feats[cols]
    y = feats[MARGIN_LABEL_COL].astype(float).to_numpy()
    reg = make_regressor(kind, config).fit(X, y)
    resid = y - reg.predict(X)
    sigma = float(np.std(resid, ddof=1))
    return MarginModel(reg, cols, sigma)


def walk_forward_cv(kind: str, config: Config, feats: pd.DataFrame,
                    cols: list[str]) -> dict[str, Any]:
    """Expanding-window CV; pooled win-prob metrics + margin error."""
    min_train = config.raw["ml"]["cv"].get("min_train_seasons", 2)
    seasons = sorted(feats["season"].unique())
    p_list, ywin_list, mu_list, ymar_list = [], [], [], []
    for i, test_season in enumerate(seasons):
        if i < min_train:
            continue
        tr = feats[feats["season"].isin(seasons[:i])]
        te = feats[feats["season"] == test_season]
        model = fit_margin_model(kind, config, tr, cols)
        mu = model.predict_margin(te)
        p_list.append(norm.cdf(mu / model.sigma))
        mu_list.append(mu)
        ywin_list.append(te[LABEL_COL].astype(int).to_numpy())
        ymar_list.append(te[MARGIN_LABEL_COL].astype(float).to_numpy())
    if not p_list:
        return {"kind": kind, "log_loss": float("inf"), "brier": float("inf"),
                "margin_mae": float("inf"), "margin_rmse": float("inf")}
    p = np.clip(np.concatenate(p_list), _EPS, 1 - _EPS)
    ywin = np.concatenate(ywin_list)
    mu = np.concatenate(mu_list)
    ymar = np.concatenate(ymar_list)
    return {
        "kind": kind,
        "log_loss": float(log_loss(ywin, p, labels=[0, 1])),
        "brier": float(brier_score_loss(ywin, p)),
        "margin_mae": float(mean_absolute_error(ymar, mu)),
        "margin_rmse": float(np.sqrt(np.mean((ymar - mu) ** 2))),
        "n": int(len(ywin)),
        "_p": p,
        "_y": ywin,
    }


def train_margin_select(config: Config, feats: pd.DataFrame,
                        kinds: tuple[str, ...] = ("ridge", "lightgbm"),
                        log: Callable[[str], None] = print) -> dict[str, Any]:
    """CV each regressor, select by derived win-prob log loss, refit on all data."""
    cols = feature_columns(feats)
    reports = []
    for kind in kinds:
        rep = walk_forward_cv(kind, config, feats, cols)
        reports.append({k: v for k, v in rep.items() if not k.startswith("_")})
        log(f"[cv-margin] {kind:>9}: log_loss={rep['log_loss']:.4f} "
            f"mae={rep['margin_mae']:.3f} rmse={rep['margin_rmse']:.3f}")
    best = min(reports, key=lambda r: r["log_loss"])
    log(f"[cv-margin] selected: {best['kind']} (log_loss={best['log_loss']:.4f})")
    model = fit_margin_model(best["kind"], config, feats, cols)
    return {"model": model, "cols": cols,
            "report": {"selected": best["kind"], "sigma": model.sigma, "cv": reports}}
