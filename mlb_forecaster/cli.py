"""Command-line interface: scrape | rate | fit-elo | train | forecast | update | backtest | compare | serve | mcp."""

from __future__ import annotations

import json
import warnings

import click

from . import pipeline
from .config import load_config
from .selfcorrect.update import update as run_update

warnings.filterwarnings("ignore", category=UserWarning)


def _seasons(ctx_config, spec: str | None):
    if spec:
        return pipeline.parse_seasons(spec)
    return sorted(ctx_config.raw["data"]["default_seasons"])


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def main(ctx, config_path):
    """ML-driven, FiveThirtyEight-style MLB forecaster."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@main.command()
@click.option("--seasons", default=None, help='e.g. "2017-2025" or "2021,2023"')
@click.option("--no-boxscores", is_flag=True, help="Skip starter box lines (fast)")
@click.option("--refresh", is_flag=True, help="Bypass cache for the schedule")
@click.pass_context
def scrape(ctx, seasons, no_boxscores, refresh):
    """Ingest schedules, results, and pitcher box lines."""
    cfg = ctx.obj["config"]
    pipeline.scrape(cfg, _seasons(cfg, seasons),
                    fetch_boxscores=not no_boxscores, refresh=refresh)


@main.command()
@click.option("--seasons", default=None)
@click.pass_context
def rate(ctx, seasons):
    """Run the Elo engine and write ratings.csv."""
    cfg = ctx.obj["config"]
    pipeline.rate(cfg, _seasons(cfg, seasons))


@main.command(name="fit-elo")
@click.option("--seasons", default=None)
@click.pass_context
def fit_elo_cmd(ctx, seasons):
    """Optimize Elo hyperparameters (persisted with the next train)."""
    cfg = ctx.obj["config"]
    _, report = pipeline.fit_elo_params(cfg, _seasons(cfg, seasons))
    click.echo(json.dumps(report, indent=2))


@main.command()
@click.option("--seasons", default=None)
@click.option("--through", type=int, default=None, help="Train through this season")
@click.option("--fit-elo", "fit_elo_first", is_flag=True,
              help="Fit Elo hyperparameters before training")
@click.pass_context
def train(ctx, seasons, through, fit_elo_first):
    """Train + select the model (logistic vs LightGBM) and save it."""
    cfg = ctx.obj["config"]
    season_list = _seasons(cfg, seasons)
    params = None
    if fit_elo_first:
        fit_seasons = [s for s in season_list if through is None or s <= through]
        params, _ = pipeline.fit_elo_params(cfg, fit_seasons)
    pipeline.train(cfg, season_list, through=through, params=params)


@main.command()
@click.option("--season", type=int, required=True)
@click.option("--sims", type=int, default=None, help="Number of Monte Carlo sims")
@click.pass_context
def forecast(ctx, season, sims):
    """Simulate the remaining season and write forecast.json."""
    cfg = ctx.obj["config"]
    odds = pipeline.forecast(cfg, season, n_sims=sims)
    cols = ["team", "league", "current_wins", "proj_wins", "proj_run_diff",
            "make_playoffs", "win_division", "win_pennant", "win_ws"]
    click.echo(odds[cols].to_string(index=False))


@main.command()
@click.option("--season", type=int, required=True)
@click.option("--retrain", is_flag=True, help="Force ML retrain")
@click.option("--refit-elo", is_flag=True, help="Force Elo hyperparameter refit")
@click.pass_context
def update(ctx, season, retrain, refit_elo):
    """Self-correct: ingest latest, advance Elo, retrain on cadence."""
    cfg = ctx.obj["config"]
    entry = run_update(cfg, season, force_retrain=retrain, force_refit_elo=refit_elo)
    click.echo(json.dumps(entry, indent=2))


@main.command()
@click.option("--seasons", default=None)
@click.option("--through", type=int, default=None)
@click.pass_context
def backtest(ctx, seasons, through):
    """Walk-forward backtest: Elo vs ML, log loss / Brier / calibration."""
    cfg = ctx.obj["config"]
    season_list = _seasons(cfg, seasons)
    if through is not None:
        season_list = [s for s in season_list if s <= through]
    report = pipeline.backtest(cfg, season_list)
    click.echo(json.dumps({k: report[k] for k in ("test_seasons", "models", "best_model")}, indent=2))
    click.echo("calibration (best model):")
    for row in report["calibration_best"]:
        click.echo(f"  {row['bucket']}: pred={row['pred_mean']:.3f} obs={row['obs_rate']:.3f} (n={row['n']})")


@main.command()
@click.option("--seasons", default=None)
@click.option("--presets", "presets_path", default="experiments.yaml",
              help="Path to the experiment presets file")
@click.option("--only", default=None, help="Comma-separated preset names to run")
@click.pass_context
def compare(ctx, seasons, presets_path, only):
    """Compare model/training variants from experiments.yaml (report only)."""
    cfg = ctx.obj["config"]
    names = [n.strip() for n in only.split(",") if n.strip()] if only else None
    report = pipeline.compare(cfg, _seasons(cfg, seasons), presets_path, names=names)
    results = report["results"]
    if not results:
        click.echo("no variants evaluated")
        return
    best_ll = results[0]["log_loss"]
    click.echo(f"\nvariants ranked by walk-forward CV log loss "
               f"(seasons {report['seasons'][0]}-{report['seasons'][-1]}):")
    click.echo(f"  {'name':<20} {'family':<9} {'log_loss':>10} {'brier':>8} "
               f"{'n':>7} {'vs-best':>9}")
    for r in results:
        click.echo(f"  {r['name']:<20} {r['family']:<9} {r['log_loss']:>10.5f} "
                   f"{r['brier']:>8.4f} {r['n']:>7} {r['log_loss'] - best_ll:>+9.5f}")
    click.echo(f"\nbest: {report['best']}  "
               f"(adopt by editing config.yaml, then `mlbfc train`)")


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--port", type=int, default=8000, help="Port")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev)")
def serve(host, port, reload):
    """Run the REST API (FastAPI) for agents/clients to call tools against."""
    import uvicorn  # imported lazily so the 'service' extra is optional

    uvicorn.run("mlb_forecaster.service.api:app", host=host, port=port, reload=reload)


@main.command()
def mcp():
    """Run the MCP server (stdio) exposing the forecaster as agent tools."""
    from .service.mcp_server import main as run_mcp  # lazy: optional 'service' extra

    run_mcp()


if __name__ == "__main__":
    main()
