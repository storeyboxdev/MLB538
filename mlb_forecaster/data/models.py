"""Core dataclasses shared across the pipeline.

Games are stored as flat CSV rows; these dataclasses document the canonical
schema and provide (de)serialization helpers. Home team is always ``*_home`` /
"team1" in 538 parlance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Optional


@dataclass
class Game:
    """A single MLB game, normalized from the Stats API."""

    game_pk: int
    date: date
    season: int
    game_type: str            # "R" regular, "F"/"D"/"L"/"W" postseason rounds, etc.
    home_team: str            # team abbreviation, e.g. "NYY"
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "Scheduled"  # "Final", "Scheduled", "In Progress", ...
    # Probable / actual starting pitchers.
    home_pitcher_id: Optional[int] = None
    away_pitcher_id: Optional[int] = None
    home_pitcher: Optional[str] = None
    away_pitcher: Optional[str] = None
    # Starter box-score lines (populated from the boxscore endpoint when Final).
    home_pitcher_line: Optional["PitcherLine"] = None
    away_pitcher_line: Optional["PitcherLine"] = None
    venue_id: Optional[int] = None
    double_header: str = "N"

    @property
    def is_final(self) -> bool:
        return self.status == "Final" and self.home_score is not None and self.away_score is not None

    @property
    def home_win(self) -> Optional[int]:
        if not self.is_final:
            return None
        return int(self.home_score > self.away_score)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["date"] = self.date.isoformat()
        # Flatten pitcher lines into prefixed columns.
        for side in ("home", "away"):
            line = row.pop(f"{side}_pitcher_line")
            if line:
                for k, v in line.items():
                    row[f"{side}_{k}"] = v
        return row


@dataclass
class PitcherLine:
    """A starter's box-score line, used to compute a Bill James game score."""

    outs: int = 0             # innings pitched expressed in outs (IP * 3)
    hits: int = 0
    runs: int = 0
    earned_runs: int = 0
    walks: int = 0
    strikeouts: int = 0
    home_runs: int = 0

    def items(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class TeamRating:
    """Current Elo state for a team (used during forecasting)."""

    team: str
    rating: float
    league: str = ""
    division: str = ""
    games_played: int = 0
    wins: int = 0
    losses: int = 0


@dataclass
class PitcherRating:
    """Rolling game-score state for a starter."""

    pitcher_id: int
    name: str = ""
    rolling_game_score: float = field(default=float("nan"))
    starts: int = 0
