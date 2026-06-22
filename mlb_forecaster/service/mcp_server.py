"""MCP front-end over :class:`Service`.

Exposes the forecaster as MCP tools so AI agents (Claude Code, Claude API agents,
etc.) can call them natively. Run with ``mlbfc mcp`` (stdio transport). Heavy
commands return a job descriptor; poll with the ``get_job`` tool.

Tool docstrings are surfaced to the agent as tool descriptions, so they are
written for that audience.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .core import DEFAULT_PRESETS_PATH, Service

mcp = FastMCP("mlb-forecaster")
_service = Service()


# --- fast reads ---------------------------------------------------------
@mcp.tool()
def get_model_info() -> dict:
    """Return metadata for the latest trained model (version, CV report, features,
    fitted Elo params), or {"available": false} if none has been trained yet."""
    return _service.model_info()


@mcp.tool()
def list_presets(presets_path: str = DEFAULT_PRESETS_PATH) -> dict:
    """List the experiment presets in experiments.yaml (names + their config overrides)."""
    return _service.list_presets(presets_path)


@mcp.tool()
def forecast(season: int, sims: Optional[int] = None) -> dict:
    """Run the season Monte Carlo simulation and return projected standings/odds
    (proj wins, run diff, playoff/division/pennant/World Series probabilities) per
    team. Synchronous; lower ``sims`` for a faster, rougher estimate."""
    return _service.forecast(season, sims=sims)


# --- background jobs (return a job; poll get_job) -----------------------
@mcp.tool()
def submit_scrape(seasons: Optional[str] = None, no_boxscores: bool = False,
                  refresh: bool = False) -> dict:
    """Start a background scrape of MLB schedules/results for ``seasons`` (e.g.
    "2017-2025" or "2021,2023"; defaults to configured seasons). Box scores are
    slow (~3 min/season); pass no_boxscores=true for a fast schedule-only pull.
    Returns a job; poll get_job for completion."""
    return _service.submit_scrape(seasons, no_boxscores, refresh)


@mcp.tool()
def submit_fit_elo(seasons: Optional[str] = None) -> dict:
    """Start a background re-optimization of the Elo hyperparameters over ``seasons``.
    Returns a job; poll get_job."""
    return _service.submit_fit_elo(seasons)


@mcp.tool()
def submit_train(seasons: Optional[str] = None, through: Optional[int] = None,
                 fit_elo: bool = False) -> dict:
    """Start a background training run (logistic vs LightGBM via walk-forward CV)
    and persist the selected model. Set fit_elo=true to re-fit Elo first. Returns
    a job; poll get_job."""
    return _service.submit_train(seasons, through, fit_elo)


@mcp.tool()
def submit_backtest(seasons: Optional[str] = None,
                    through: Optional[int] = None) -> dict:
    """Start a background walk-forward backtest comparing Elo baselines vs the ML
    models (log loss / Brier / calibration). Returns a job; poll get_job."""
    return _service.submit_backtest(seasons, through)


@mcp.tool()
def submit_compare(seasons: Optional[str] = None,
                   presets_path: str = DEFAULT_PRESETS_PATH,
                   only: Optional[str] = None) -> dict:
    """Start a background comparison of experiment presets (from experiments.yaml)
    under walk-forward CV. ``only`` is a comma-separated subset of preset names.
    Report-only. Returns a job; poll get_job."""
    return _service.submit_compare(seasons, presets_path, only)


@mcp.tool()
def submit_update(season: int, retrain: bool = False, refit_elo: bool = False) -> dict:
    """Start a background self-correction cycle for ``season``: ingest latest
    results, advance Elo, and retrain on cadence (force with retrain/refit_elo).
    Returns a job; poll get_job."""
    return _service.submit_update(season, retrain, refit_elo)


# --- job inspection -----------------------------------------------------
@mcp.tool()
def get_job(job_id: str) -> dict:
    """Get the status, logs, and (when finished) result of a background job."""
    job = _service.get_job(job_id)
    return job or {"error": f"No such job: {job_id}"}


@mcp.tool()
def list_jobs() -> dict:
    """List all background jobs (most recent first), without their full results."""
    return {"jobs": _service.list_jobs()}


def main() -> None:
    """Entry point for ``mlbfc mcp`` — serve over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
