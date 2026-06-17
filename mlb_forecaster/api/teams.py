"""Team metadata: abbreviation, league, division."""

from __future__ import annotations

from typing import Any

from .client import MLBStatsClient

# League ids per MLB Stats API.
LEAGUE_NAMES = {103: "AL", 104: "NL"}


def fetch_teams(client: MLBStatsClient, season: int) -> dict[int, dict[str, Any]]:
    """Return ``{team_id: {abbr, name, league, division}}`` for MLB teams."""
    payload = client.get("v1/teams", {"sportId": 1, "season": season})
    out: dict[int, dict[str, Any]] = {}
    for t in payload["teams"]:
        league_id = t.get("league", {}).get("id")
        out[t["id"]] = {
            "abbr": t.get("abbreviation"),
            "name": t.get("name"),
            "league": LEAGUE_NAMES.get(league_id, ""),
            "league_id": league_id,
            "division": t.get("division", {}).get("name", ""),
            "division_id": t.get("division", {}).get("id"),
        }
    return out
