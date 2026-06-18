"""Shared pipeline steps used by the CLI and the self-correction loop."""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from .api.client import MLBStatsClient
from .api.games import ingest_season
from .api.teams import fetch_teams
from .backtest.evaluate import run_backtest
from .config import Config
from .data import gcs, store
from .elo.engine import final_ratings, run_engine
from .elo.fit import fit_elo
from .elo.params import EloParams
from .forecast.odds import write_forecast
from .forecast.simulate import simulate_season
from .ml.features import build_features
from .ml.margin import train_margin_select
from .ml.predict import Predictor
from .ml.registry import load_elo_params, load_model, save_model
from .ml.train import train_select

Logger = Callable[[str], None]


def parse_seasons(spec: str) -> list[int]:
    """Parse "2017-2025" or "2017,2019,2021" into a sorted list of seasons."""
    seasons: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            seasons.update(range(int(lo), int(hi) + 1))
        else:
            seasons.add(int(part))
    return sorted(seasons)


def get_params(config: Config) -> EloParams:
    """Use fitted Elo params from the registry if present, else config defaults."""
    fitted = load_elo_params(config.models_dir, "latest")
    return fitted or EloParams.from_config(config)


def get_team_meta(config: Config, season: int) -> dict[str, dict]:
    """Return ``{abbr: {league, division, ...}}`` for a season."""
    client = MLBStatsClient(config)
    return {m["abbr"]: m for m in fetch_teams(client, season).values()}


# --- pipeline steps -----------------------------------------------------
def scrape(config: Config, seasons: list[int], *, fetch_boxscores: bool = True,
           refresh: bool = False, log: Logger = print) -> None:
    """Ingest each season and write processed game logs."""
    config.ensure_dirs()
    for s in seasons:
        df = ingest_season(config, s, fetch_boxscores=fetch_boxscores,
                           use_cache=not refresh)
        path = store.write_games(df, config.processed_dir, s)
        msg = (f"[scrape] {s}: {len(df)} games -> {path.name} "
               f"({int((df['status'] == 'Final').sum())} final)")
        if gcs.is_enabled(config):
            uri = gcs.upload_games(df, config, s)
            msg += f" -> {uri}"
        log(msg)


def load_games(config: Config, seasons: list[int]) -> pd.DataFrame:
    return store.read_many_seasons(config.processed_dir, seasons)


def rate(config: Config, seasons: list[int], params: Optional[EloParams] = None,
         log: Logger = print) -> pd.DataFrame:
    """Run the Elo engine over all seasons; write ratings.csv; return engine df."""
    params = params or get_params(config)
    games = load_games(config, seasons)
    eng = run_engine(games, params)
    store.write_csv(eng, config.output_dir / "ratings.csv")
    msg = f"[rate] {len(eng)} game rows -> ratings.csv"
    if gcs.is_enabled(config):
        msg += f" -> {gcs.upload_ratings(eng, config)}"
    log(msg)
    return eng


def fit_elo_params(config: Config, seasons: list[int], log: Logger = print
                   ) -> tuple[EloParams, dict[str, Any]]:
    """Fit Elo hyperparameters on the given seasons."""
    games = load_games(config, seasons)
    base = EloParams.from_config(config)
    best, report = fit_elo(games, base, config.raw["elo"]["fit"])
    log(f"[fit-elo] log_loss {report['baseline_log_loss']:.4f} -> "
        f"{report['fitted_log_loss']:.4f}; fitted={report['fitted']}")
    return best, report


def train(config: Config, seasons: list[int], through: Optional[int] = None,
          params: Optional[EloParams] = None, log: Logger = print) -> dict[str, Any]:
    """Build features and train/select the model; persist to the registry."""
    params = params or get_params(config)
    if through is not None:
        seasons = [s for s in seasons if s <= through]
    games = load_games(config, seasons)
    eng = run_engine(games, params)
    feats = build_features(eng, games, config)
    result = train_select(config, feats, log=log)

    margin_model = None
    report = result["report"]
    if config.raw["ml"].get("margin", {}).get("enabled", True):
        margin_result = train_margin_select(config, feats, log=log)
        margin_model = margin_result["model"]
        report = {**report, "margin": margin_result["report"]}

    dest = save_model(config.models_dir, result["model"], result["cols"],
                      params, report, margin_model=margin_model)
    log(f"[train] saved model -> {dest}")
    return report


def forecast(config: Config, season: int, n_sims: Optional[int] = None,
             seasons_for_ratings: Optional[list[int]] = None,
             log: Logger = print) -> pd.DataFrame:
    """Compute current ratings, predict remaining games, simulate the season."""
    params = get_params(config)
    seasons = seasons_for_ratings or [s for s in parse_seasons(
        f"{min(config.raw['data']['default_seasons'])}-{season}")
        if (config.processed_dir / f"games_{s}.csv").exists()]
    if season not in seasons and (config.processed_dir / f"games_{season}.csv").exists():
        seasons.append(season)
    seasons = sorted(set(seasons))

    games_all = load_games(config, seasons)
    eng = run_engine(games_all, params)
    ratings = final_ratings(eng)

    games_season = store.read_games(config.processed_dir, season)
    team_meta = get_team_meta(config, season)

    # Use the trained model if available; else fall back to Elo probability.
    prob_fn = None
    margin_fn = None
    model_version = "elo_fallback"
    source = config.raw["forecast"].get("win_prob_source", "classifier")
    try:
        bundle = load_model(config.models_dir, "latest")
        predictor = Predictor(bundle["model"], bundle["cols"])
        prob_fn = predictor.predict_home_prob
        model_version = bundle["metadata"].get("version", "latest")
        margin_model = bundle.get("margin")
        if margin_model is not None:
            margin_fn = margin_model.predict_margin
            if source == "margin":
                prob_fn = margin_model.predict_proba_home
                model_version += "+margin"
    except FileNotFoundError:
        log("[forecast] no trained model found; using Elo probability fallback")

    odds = simulate_season(games_season, ratings, team_meta, config,
                           prob_fn=prob_fn, margin_fn=margin_fn, n_sims=n_sims)
    n = n_sims or config.raw["forecast"]["n_sims"]
    path = write_forecast(odds, config.output_dir, season, n, model_version)
    msg = f"[forecast] {season}: wrote {path.name} ({n} sims, model={model_version})"
    if gcs.is_enabled(config):
        msg += f" -> {gcs.upload_forecast(odds, config, season, model_version)}"
    log(msg)
    return odds


def backtest(config: Config, seasons: list[int], params: Optional[EloParams] = None,
             log: Logger = print) -> dict[str, Any]:
    """Run the walk-forward backtest and persist metrics.json."""
    params = params or get_params(config)
    games = load_games(config, seasons)
    report = run_backtest(config, games, params)
    store.write_json(report, config.output_dir / "backtest.json")
    log(f"[backtest] best={report['best_model']} | " + " | ".join(
        f"{k}={v['log_loss']:.4f}" for k, v in report["models"].items()))
    return report
