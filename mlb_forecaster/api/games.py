"""Boxscore enrichment (starter pitching lines) and season ingestion."""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..data.models import Game, PitcherLine
from .client import MLBStatsClient
from .schedule import POSTSEASON_TYPES, fetch_schedule
from .teams import fetch_teams


def _parse_starter_line(client: MLBStatsClient, game_pk: int) -> dict[str, tuple[int, PitcherLine]]:
    """Return ``{"home": (starter_id, PitcherLine), ...}`` for the starters."""
    box = client.get(f"v1/game/{game_pk}/boxscore")
    out: dict[str, tuple[int, PitcherLine]] = {}
    for side in ("home", "away"):
        team = box["teams"][side]
        pitchers = team.get("pitchers", [])
        if not pitchers:
            continue
        starter_id = pitchers[0]  # first pitcher used = the starter
        pdata = team["players"].get(f"ID{starter_id}")
        if not pdata:
            continue
        ps = pdata.get("stats", {}).get("pitching", {})
        if not ps:
            continue
        line = PitcherLine(
            outs=int(ps.get("outs", 0) or 0),
            hits=int(ps.get("hits", 0) or 0),
            runs=int(ps.get("runs", 0) or 0),
            earned_runs=int(ps.get("earnedRuns", 0) or 0),
            walks=int(ps.get("baseOnBalls", 0) or 0),
            strikeouts=int(ps.get("strikeOuts", 0) or 0),
            home_runs=int(ps.get("homeRuns", 0) or 0),
        )
        out[side] = (starter_id, line)
    return out


def ingest_season(config: Config, season: int, *, fetch_boxscores: bool = True,
                  use_cache: bool = True) -> pd.DataFrame:
    """Ingest one season into a normalized DataFrame and return it.

    Boxscore enrichment (starter lines) is needed for pitcher game scores; it can
    be skipped with ``fetch_boxscores=False`` for a fast schedule-only pull.
    """
    client = MLBStatsClient(config)
    teams = fetch_teams(client, season)
    abbr_by_id = {tid: meta["abbr"] for tid, meta in teams.items()}

    games = fetch_schedule(client, season, abbr_by_id, use_cache=use_cache)

    starter_ids: dict[int, dict[str, int]] = {}
    if fetch_boxscores:
        finals = [g for g in games if g.is_final]
        for g in tqdm(finals, desc=f"boxscores {season}", unit="game"):
            try:
                lines = _parse_starter_line(client, g.game_pk)
            except RuntimeError:
                continue
            ids: dict[str, int] = {}
            for side in ("home", "away"):
                if side in lines:
                    sid, line = lines[side]
                    setattr(g, f"{side}_pitcher_line", line)
                    ids[side] = sid
            starter_ids[g.game_pk] = ids

    rows = []
    for g in games:
        row = g.to_row()
        row["playoff"] = g.game_type in POSTSEASON_TYPES
        ids = starter_ids.get(g.game_pk, {})
        # Actual starter ids (from boxscore) when available, else probable ids.
        row["home_starter_id"] = ids.get("home", g.home_pitcher_id)
        row["away_starter_id"] = ids.get("away", g.away_pitcher_id)
        rows.append(row)
    df = pd.DataFrame(rows)

    # The schedule sometimes returns a gamePk twice (a played row + a postponed/
    # split-doubleheader placeholder with no score). Keep the played row.
    df["_has_score"] = df["home_score"].notna()
    df = (df.sort_values(["game_pk", "_has_score"], ascending=[True, False])
            .drop_duplicates("game_pk", keep="first")
            .drop(columns="_has_score")
            .sort_values(["date", "game_pk"])
            .reset_index(drop=True))
    return df
