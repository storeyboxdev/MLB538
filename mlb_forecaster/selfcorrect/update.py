"""Self-correction: ingest latest results, advance Elo, retrain on cadence.

Online Elo correction is inherent: re-running the engine over the refreshed game
logs recomputes ratings forward through the newest finals. On top of that, the ML
layer is retrained (and Elo hyperparameters re-optimized) on a rolling window
whenever the configured cadence has elapsed, and rolling error is logged so drift
is visible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..config import Config
from ..data import store
from ..elo.engine import run_engine
from ..ml.features import build_features
from ..ml.predict import Predictor
from ..ml.registry import load_model
from .. import pipeline

Logger = Callable[[str], None]
METRICS_FILE = "metrics.json"


def _load_metrics(path: Path) -> list[dict]:
    if path.exists():
        return store.read_json(path)
    return []


def _days_since(iso: Optional[str]) -> float:
    if not iso:
        return float("inf")
    then = datetime.fromisoformat(iso)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 86400.0


def _rolling_window_seasons(config: Config, season: int) -> list[int]:
    n = config.raw["selfcorrect"].get("rolling_train_seasons", 5)
    avail = [s for s in pipeline.parse_seasons(
        f"{min(config.raw['data']['default_seasons'])}-{season}")
        if (config.processed_dir / f"games_{s}.csv").exists()]
    return avail[-n:] if avail else [season]


def update(config: Config, season: int, *, force_retrain: bool = False,
           force_refit_elo: bool = False, recent_games: int = 200,
           log: Logger = print) -> dict[str, Any]:
    """Run one self-correction cycle for ``season``."""
    config.ensure_dirs()
    metrics_path = config.output_dir / METRICS_FILE
    history = _load_metrics(metrics_path)
    last = history[-1] if history else {}

    sc = config.raw["selfcorrect"]
    last_retrain = last.get("retrained_at")
    last_elo_refit = last.get("elo_refit_at")
    do_retrain = force_retrain or _days_since(last_retrain) >= sc.get("retrain_every_days", 7)
    do_refit_elo = force_refit_elo or _days_since(last_elo_refit) >= sc.get("refit_elo_every_days", 30)
    # Refitting Elo implies retraining the ML layer on the new features.
    if do_refit_elo:
        do_retrain = True

    # 1. Ingest the latest results (force-refresh the live schedule).
    pipeline.scrape(config, [season], fetch_boxscores=True, refresh=True, log=log)

    window = _rolling_window_seasons(config, season)
    log(f"[update] rolling window seasons: {window}")

    # 2. Online Elo: recompute forward through newest finals.
    params = pipeline.get_params(config)
    elo_refit_at = last_elo_refit
    if do_refit_elo:
        params, _ = pipeline.fit_elo_params(config, window, log=log)
        elo_refit_at = datetime.now(timezone.utc).isoformat()

    retrained_at = last_retrain
    if do_retrain:
        pipeline.train(config, window, params=params, log=log)
        retrained_at = datetime.now(timezone.utc).isoformat()
    else:
        log("[update] retrain cadence not reached; reusing existing model")

    # 3. Rolling error on the most recent finals using the current model.
    rolling = _rolling_metrics(config, season, params, recent_games)
    if rolling:
        log(f"[update] rolling ({rolling['n']} recent finals): "
            f"log_loss={rolling['log_loss']:.4f} brier={rolling['brier']:.4f}")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "retrained": do_retrain,
        "retrained_at": retrained_at,
        "elo_refit": do_refit_elo,
        "elo_refit_at": elo_refit_at,
        "rolling": rolling,
    }
    history.append(entry)
    store.write_json(history, metrics_path)
    return entry


def _rolling_metrics(config: Config, season: int, params, recent_games: int
                     ) -> Optional[dict]:
    """Log loss / Brier of the current model on the most recent finals."""
    try:
        bundle = load_model(config.models_dir, "latest")
    except FileNotFoundError:
        return None
    predictor = Predictor(bundle["model"], bundle["cols"])

    games = store.read_games(config.processed_dir, season)
    eng = run_engine(games, params)
    feats = build_features(eng, games, config)
    if feats.empty:
        return None
    feats = feats.sort_values(["date", "game_pk"]).tail(recent_games)
    p = predictor.predict_home_prob(feats)
    y = feats["home_win"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    return {"n": int(len(y)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p))}
