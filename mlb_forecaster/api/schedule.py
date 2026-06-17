"""Fetch and parse the season schedule (results + probable pitchers)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from ..data.models import Game
from .client import MLBStatsClient

# Regular season + postseason game types we keep. (S=spring, E=exhibition skipped.)
KEEP_GAME_TYPES = {"R", "F", "D", "L", "W"}
POSTSEASON_TYPES = {"F", "D", "L", "W"}


def _parse_game(g: dict[str, Any], abbr_by_id: dict[int, str]) -> Optional[Game]:
    game_type = g.get("gameType")
    if game_type not in KEEP_GAME_TYPES:
        return None

    home = g["teams"]["home"]
    away = g["teams"]["away"]
    home_id = home["team"]["id"]
    away_id = away["team"]["id"]
    # Skip non-MLB opponents (e.g. exhibitions vs minor-league clubs).
    if home_id not in abbr_by_id or away_id not in abbr_by_id:
        return None

    official = g.get("officialDate") or g["gameDate"][:10]
    game_date = datetime.strptime(official, "%Y-%m-%d").date()
    status = g["status"].get("abstractGameState", "Scheduled")

    home_pp = home.get("probablePitcher") or {}
    away_pp = away.get("probablePitcher") or {}

    return Game(
        game_pk=g["gamePk"],
        date=game_date,
        season=int(g.get("season", game_date.year)),
        game_type=game_type,
        home_team=abbr_by_id[home_id],
        away_team=abbr_by_id[away_id],
        home_score=home.get("score"),
        away_score=away.get("score"),
        status="Final" if status == "Final" else status,
        home_pitcher_id=home_pp.get("id"),
        away_pitcher_id=away_pp.get("id"),
        home_pitcher=home_pp.get("fullName"),
        away_pitcher=away_pp.get("fullName"),
        venue_id=g.get("venue", {}).get("id"),
        double_header=g.get("doubleHeader", "N"),
    )


def fetch_schedule(client: MLBStatsClient, season: int,
                   abbr_by_id: dict[int, str],
                   use_cache: bool = True) -> list[Game]:
    """Fetch all kept games for a season in one schedule request."""
    params = {
        "sportId": 1,
        "startDate": f"{season}-03-01",
        "endDate": f"{season}-11-30",
        "hydrate": "probablePitcher,linescore,team",
    }
    payload = client.get("v1/schedule", params, use_cache=use_cache)
    games: list[Game] = []
    for d in payload.get("dates", []):
        for g in d.get("games", []):
            game = _parse_game(g, abbr_by_id)
            if game is not None:
                games.append(game)
    games.sort(key=lambda g: (g.date, g.game_pk))
    return games
