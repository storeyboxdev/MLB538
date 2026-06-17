"""Format and persist forecast odds to JSON."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..data.store import write_json


def odds_to_payload(odds: pd.DataFrame, season: int, n_sims: int,
                    model_version: str) -> dict:
    """Convert the odds DataFrame into a JSON-serializable payload."""
    teams = []
    for _, r in odds.iterrows():
        teams.append({
            "team": r["team"],
            "league": r["league"],
            "division": r["division"],
            "current_wins": int(r["current_wins"]),
            "proj_wins": float(r["proj_wins"]),
            "proj_run_diff": float(r["proj_run_diff"]),
            "make_playoffs": round(float(r["make_playoffs"]), 4),
            "win_division": round(float(r["win_division"]), 4),
            "win_pennant": round(float(r["win_pennant"]), 4),
            "win_ws": round(float(r["win_ws"]), 4),
        })
    return {
        "season": season,
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_sims": n_sims,
        "model_version": model_version,
        "teams": teams,
    }


def write_forecast(odds: pd.DataFrame, output_dir: Path, season: int,
                   n_sims: int, model_version: str) -> Path:
    payload = odds_to_payload(odds, season, n_sims, model_version)
    return write_json(payload, output_dir / "forecast.json")
