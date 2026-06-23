"""Shared service core wrapping the pipeline for both front-ends.

Fast methods return immediately. ``submit_*`` methods enqueue a background job
(see :class:`JobManager`) and return the job descriptor; poll :meth:`get_job`.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import pipeline
from ..config import Config, load_config
from ..experiments import load_presets
from ..ml.registry import load_model
from ..selfcorrect.update import update as run_update
from .jobs import Job, JobManager

DEFAULT_PRESETS_PATH = "experiments.yaml"
_FORECAST_COLS = ["team", "league", "current_wins", "proj_wins", "proj_run_diff",
                  "make_playoffs", "win_division", "win_pennant", "win_ws"]


class Service:
    """The single object both the REST and MCP front-ends call into."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or load_config()
        self.jobs = JobManager()

    # --- helpers --------------------------------------------------------
    def _seasons(self, spec: Optional[str]) -> list[int]:
        if spec:
            return pipeline.parse_seasons(spec)
        return sorted(self.config.raw["data"]["default_seasons"])

    # --- fast / synchronous --------------------------------------------
    def model_info(self) -> dict[str, Any]:
        """Metadata for the latest persisted model, or availability flag."""
        try:
            bundle = load_model(self.config.models_dir, "latest")
        except FileNotFoundError:
            return {"available": False}
        meta = bundle.get("metadata", {})
        return {
            "available": True,
            "version": meta.get("version"),
            "created": meta.get("created"),
            "report": meta.get("report"),
            "feature_columns": bundle.get("cols"),
            "elo_params": bundle["elo_params"].to_dict(),
            "has_margin_model": bundle.get("margin") is not None,
        }

    def list_presets(self, presets_path: str = DEFAULT_PRESETS_PATH) -> dict[str, Any]:
        """Names + overrides of the experiment presets in ``experiments.yaml``."""
        presets = load_presets(self._resolve(presets_path))
        return {"presets": presets, "names": list(presets)}

    def forecast(self, season: int, sims: Optional[int] = None) -> dict[str, Any]:
        """Run the season simulation synchronously and return projected odds."""
        odds = pipeline.forecast(self.config, season, n_sims=sims)
        cols = [c for c in _FORECAST_COLS if c in odds.columns]
        return {
            "season": season,
            "n_sims": sims or self.config.raw["forecast"]["n_sims"],
            "teams": odds[cols].to_dict(orient="records"),
        }

    # --- heavy / background --------------------------------------------
    def submit_scrape(self, seasons: Optional[str] = None,
                      no_boxscores: bool = False, refresh: bool = False) -> dict:
        season_list = self._seasons(seasons)

        def body(log):
            pipeline.scrape(self.config, season_list, fetch_boxscores=not no_boxscores,
                            refresh=refresh, log=log)
            return {"scraped_seasons": season_list}

        return self.jobs.submit("scrape", body,
                                {"seasons": season_list, "no_boxscores": no_boxscores}).to_dict()

    def submit_fit_elo(self, seasons: Optional[str] = None) -> dict:
        season_list = self._seasons(seasons)

        def body(log):
            _, report = pipeline.fit_elo_params(self.config, season_list, log=log)
            return report

        return self.jobs.submit("fit-elo", body, {"seasons": season_list}).to_dict()

    def submit_train(self, seasons: Optional[str] = None, through: Optional[int] = None,
                     fit_elo: bool = False) -> dict:
        season_list = self._seasons(seasons)

        def body(log):
            params = None
            if fit_elo:
                fit_seasons = [s for s in season_list if through is None or s <= through]
                params, _ = pipeline.fit_elo_params(self.config, fit_seasons, log=log)
            return pipeline.train(self.config, season_list, through=through,
                                  params=params, log=log)

        return self.jobs.submit("train", body,
                                {"seasons": season_list, "through": through,
                                 "fit_elo": fit_elo}).to_dict()

    def submit_backtest(self, seasons: Optional[str] = None,
                        through: Optional[int] = None) -> dict:
        season_list = self._seasons(seasons)
        if through is not None:
            season_list = [s for s in season_list if s <= through]

        def body(log):
            return pipeline.backtest(self.config, season_list, log=log)

        return self.jobs.submit("backtest", body, {"seasons": season_list}).to_dict()

    def submit_compare(self, seasons: Optional[str] = None,
                       presets_path: str = DEFAULT_PRESETS_PATH,
                       only: Optional[str] = None) -> dict:
        season_list = self._seasons(seasons)
        names = [n.strip() for n in only.split(",") if n.strip()] if only else None

        def body(log):
            return pipeline.compare(self.config, season_list,
                                    self._resolve(presets_path), names=names, log=log)

        return self.jobs.submit("compare", body,
                                {"seasons": season_list, "only": names}).to_dict()

    def submit_update(self, season: int, retrain: bool = False,
                      refit_elo: bool = False) -> dict:
        def body(log):
            return run_update(self.config, season, force_retrain=retrain,
                              force_refit_elo=refit_elo, log=log)

        return self.jobs.submit("update", body,
                                {"season": season, "retrain": retrain,
                                 "refit_elo": refit_elo}).to_dict()

    # --- job inspection -------------------------------------------------
    def get_job(self, job_id: str) -> Optional[dict]:
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None

    def list_jobs(self) -> list[dict]:
        return [j.to_dict(include_result=False) for j in self.jobs.list()]

    # --- internal -------------------------------------------------------
    def _resolve(self, path: str) -> str:
        """Resolve a possibly-relative path against the config repo root."""
        from pathlib import Path

        p = Path(path)
        return str(p if p.is_absolute() else self.config.root / p)
