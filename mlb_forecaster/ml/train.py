"""Train and select the prediction model with walk-forward CV.

Two model families are evaluated under expanding-window cross-validation by
season (train on seasons < Y, test on Y): regularized logistic regression and
LightGBM. The lower-log-loss family is refit on all available data (with
calibration) and returned for persistence.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config
from .calibrate import CalibratedModel, fit_calibrator
from .features import LABEL_COL, feature_columns


def make_estimator(kind: str, config: Config):
    """Build an uncalibrated estimator for the given family."""
    if kind == "logistic":
        p = config.raw["ml"]["logistic"]
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=p.get("C", 1.0),
                                       max_iter=p.get("max_iter", 1000))),
        ])
    if kind == "lightgbm":
        p = config.raw["ml"]["lightgbm"]
        return LGBMClassifier(
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
    raise ValueError(f"Unknown model kind: {kind}")


def fit_calibrated_model(kind: str, config: Config, feats: pd.DataFrame,
                         cols: list[str], method: str,
                         calib_frac: float = 0.2) -> CalibratedModel:
    """Fit base estimator on the earlier slice, calibrator on the held-out tail."""
    feats = feats.sort_values(["date", "game_pk"]).reset_index(drop=True)
    y = feats[LABEL_COL].astype(int).to_numpy()
    X = feats[cols]

    if method in (None, "none") or len(feats) < 200:
        est = make_estimator(kind, config).fit(X, y)
        return CalibratedModel(est, cols, method="none", calibrator=None)

    n_cal = max(50, int(len(feats) * calib_frac))
    X_base, y_base = X.iloc[:-n_cal], y[:-n_cal]
    X_cal, y_cal = X.iloc[-n_cal:], y[-n_cal:]
    est = make_estimator(kind, config).fit(X_base, y_base)
    raw = est.predict_proba(X_cal)[:, 1]
    calibrator = fit_calibrator(method, raw, y_cal)
    return CalibratedModel(est, cols, method=method, calibrator=calibrator)


def walk_forward_cv(kind: str, config: Config, feats: pd.DataFrame,
                    cols: list[str], method: str) -> dict[str, Any]:
    """Expanding-window CV by season; returns pooled out-of-sample metrics."""
    min_train = config.raw["ml"]["cv"].get("min_train_seasons", 2)
    seasons = sorted(feats["season"].unique())
    preds, labels, fold_rows = [], [], []
    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < min_train:
            continue
        tr = feats[feats["season"].isin(train_seasons)]
        te = feats[feats["season"] == test_season]
        model = fit_calibrated_model(kind, config, tr, cols, method)
        p = model.predict_proba_home(te)
        y = te[LABEL_COL].astype(int).to_numpy()
        preds.append(p)
        labels.append(y)
        fold_rows.append({
            "test_season": int(test_season),
            "n": int(len(te)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
        })
    if not preds:
        return {"kind": kind, "folds": [], "log_loss": float("inf"), "brier": float("inf")}
    p_all = np.concatenate(preds)
    y_all = np.concatenate(labels)
    return {
        "kind": kind,
        "calibration": method,
        "folds": fold_rows,
        "log_loss": float(log_loss(y_all, p_all, labels=[0, 1])),
        "brier": float(brier_score_loss(y_all, p_all)),
        "n": int(len(y_all)),
        # Pooled out-of-sample predictions (keys with leading underscore are
        # stripped before JSON serialization).
        "_p": p_all,
        "_y": y_all,
    }


def train_select(config: Config, feats: pd.DataFrame,
                 kinds: tuple[str, ...] = ("logistic", "lightgbm"),
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run CV for each ml model, pick the best, refit on all data with calibration.

    Returns a dict with the selected ``model`` (CalibratedModel), ``cols``, and
    a ``report`` with per-family CV metrics.
    """
    method = config.raw["ml"].get("calibration", "isotonic")
    cols = feature_columns(feats)

    cv_reports = []
    for kind in kinds:
        rep = walk_forward_cv(kind, config, feats, cols, method)
        # Drop pooled prediction arrays (underscore keys) so reports stay JSON-safe.
        cv_reports.append({k: v for k, v in rep.items() if not k.startswith("_")})
        log(f"[cv] {kind:>9}: log_loss={rep['log_loss']:.4f} brier={rep['brier']:.4f} "
            f"(n={rep.get('n', 0)})")

    best = min(cv_reports, key=lambda r: r["log_loss"])
    log(f"[cv] selected: {best['kind']} (log_loss={best['log_loss']:.4f})")

    final_model = fit_calibrated_model(best["kind"], config, feats, cols, method)
    return {
        "model": final_model,
        "cols": cols,
        "report": {
            "selected": best["kind"],
            "calibration": method,
            "cv": cv_reports,
            "feature_columns": cols,
        },
    }
