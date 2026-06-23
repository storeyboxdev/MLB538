"""FastAPI front-end over :class:`Service`.

Run it with ``mlbfc serve`` (or ``uvicorn mlb_forecaster.service.api:app``).
Fast reads/forecast are synchronous; heavy commands return a job you poll at
``GET /jobs/{id}``. Interactive docs are served at ``/docs``.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .core import DEFAULT_PRESETS_PATH, Service


# --- request bodies -----------------------------------------------------
class ForecastRequest(BaseModel):
    season: int
    sims: Optional[int] = None


class ScrapeRequest(BaseModel):
    seasons: Optional[str] = None
    no_boxscores: bool = False
    refresh: bool = False


class TrainRequest(BaseModel):
    seasons: Optional[str] = None
    through: Optional[int] = None
    fit_elo: bool = False


class SeasonsRequest(BaseModel):
    seasons: Optional[str] = None
    through: Optional[int] = None


class CompareRequest(BaseModel):
    seasons: Optional[str] = None
    presets_path: str = DEFAULT_PRESETS_PATH
    only: Optional[str] = None


class UpdateRequest(BaseModel):
    season: int
    retrain: bool = False
    refit_elo: bool = False


def create_app(service: Optional[Service] = None) -> FastAPI:
    svc = service or Service()
    app = FastAPI(title="MLB Forecaster API", version="0.1.0",
                  description="Elo-feature ML MLB forecaster: forecasts, training, "
                              "and experiments. Heavy commands run as background jobs.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # --- fast reads -----------------------------------------------------
    @app.get("/model")
    def model_info() -> dict:
        return svc.model_info()

    @app.get("/presets")
    def presets(presets_path: str = DEFAULT_PRESETS_PATH) -> dict:
        return svc.list_presets(presets_path)

    @app.post("/forecast")
    def forecast(req: ForecastRequest) -> dict:
        return svc.forecast(req.season, sims=req.sims)

    # --- background jobs ------------------------------------------------
    @app.post("/jobs/scrape")
    def scrape(req: ScrapeRequest) -> dict:
        return svc.submit_scrape(req.seasons, req.no_boxscores, req.refresh)

    @app.post("/jobs/fit-elo")
    def fit_elo(req: SeasonsRequest) -> dict:
        return svc.submit_fit_elo(req.seasons)

    @app.post("/jobs/train")
    def train(req: TrainRequest) -> dict:
        return svc.submit_train(req.seasons, req.through, req.fit_elo)

    @app.post("/jobs/backtest")
    def backtest(req: SeasonsRequest) -> dict:
        return svc.submit_backtest(req.seasons, req.through)

    @app.post("/jobs/compare")
    def compare(req: CompareRequest) -> dict:
        return svc.submit_compare(req.seasons, req.presets_path, req.only)

    @app.post("/jobs/update")
    def update(req: UpdateRequest) -> dict:
        return svc.submit_update(req.season, req.retrain, req.refit_elo)

    @app.get("/jobs")
    def list_jobs() -> dict:
        return {"jobs": svc.list_jobs()}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = svc.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
        return job

    return app


app = create_app()
