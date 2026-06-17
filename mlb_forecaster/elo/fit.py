"""Fit Elo hyperparameters to history by minimizing predictive log loss.

We treat the Elo engine as a parametric model and search the configured
hyperparameters (K, HFA, MOV shape/damping, pitcher weight, reversion) to
minimize log loss of the pitcher-adjusted win probability on final games.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .engine import run_engine
from .params import EloParams

_EPS = 1e-12


def _log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, _EPS, 1 - _EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def evaluate_params(games: pd.DataFrame, params: EloParams) -> float:
    """Run the engine and return log loss on final games (pitcher-adjusted prob)."""
    out = run_engine(games, params)
    finals = out[out["home_win"].notna()]
    if finals.empty:
        return float("inf")
    return _log_loss(finals["rating_prob1"].to_numpy(), finals["home_win"].to_numpy())


def fit_elo(games: pd.DataFrame, base_params: EloParams,
            fit_config: dict[str, Any]) -> tuple[EloParams, dict[str, Any]]:
    """Optimize the hyperparameters listed in ``fit_config`` (excluding maxiter).

    Returns the best-fit params and a small report dict.
    """
    valid_fields = {f.name for f in dataclasses.fields(EloParams)}
    names = [k for k in fit_config if k != "maxiter" and k in valid_fields]
    bounds = [tuple(fit_config[k]) for k in names]
    maxiter = int(fit_config.get("maxiter", 200))

    base = base_params.to_dict()
    los = np.array([b[0] for b in bounds], dtype=float)
    his = np.array([b[1] for b in bounds], dtype=float)
    spans = np.where(his > los, his - los, 1.0)

    # Optimize in normalized [0, 1] space so a single finite-difference step is
    # meaningful for all parameters despite their very different scales (e.g.
    # mov_autocorr ~0.001 vs home_field_adv ~24).
    def to_real(u: np.ndarray) -> np.ndarray:
        return los + np.clip(u, 0.0, 1.0) * spans

    x0_real = np.array([min(max(base[n], lo), hi)
                        for n, (lo, hi) in zip(names, bounds)], dtype=float)
    u0 = (x0_real - los) / spans

    def make_params(u: np.ndarray) -> EloParams:
        real = to_real(u)
        overrides = {name: float(val) for name, val in zip(names, real)}
        return replace(base_params, **overrides)

    def objective(u: np.ndarray) -> float:
        return evaluate_params(games, make_params(u))

    baseline_loss = objective(u0)
    result = minimize(
        objective, u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(names),
        options={"maxiter": maxiter, "eps": 1e-2},
    )
    best_real = to_real(result.x)
    best_params = make_params(result.x)
    report = {
        "fitted": {name: float(val) for name, val in zip(names, best_real)},
        "baseline_log_loss": baseline_loss,
        "fitted_log_loss": float(result.fun),
        "success": bool(result.success),
        "n_iter": int(result.nit),
    }
    return best_params, report
