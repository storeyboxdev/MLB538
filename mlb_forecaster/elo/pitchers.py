"""Starting-pitcher ratings: Bill James game score + rolling tracker.

Each start is scored with the classic Bill James game score; a recency-weighted
rolling average of a starter's recent scores becomes a per-game Elo adjustment to
that team's effective rating.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from ..data.models import PitcherLine
from .params import EloParams


def game_score(line: PitcherLine) -> float:
    """Classic Bill James game score.

    Start at 50; +1 per out, +2 per inning completed after the 4th, +1 per K,
    -2 per hit, -4 per earned run, -2 per unearned run, -1 per walk.
    """
    innings_completed = line.outs // 3
    bonus_innings = max(0, innings_completed - 4)
    unearned = max(0, line.runs - line.earned_runs)
    score = 50.0
    score += line.outs            # +1 per out
    score += 2 * bonus_innings
    score += line.strikeouts
    score -= 2 * line.hits
    score -= 4 * line.earned_runs
    score -= 2 * unearned
    score -= line.walks
    return score


class PitcherTracker:
    """Maintains recency-weighted rolling game scores per pitcher id."""

    def __init__(self, params: EloParams):
        self.params = params
        self._history: dict[int, deque[float]] = {}

    def rolling_score(self, pitcher_id: Optional[int]) -> Optional[float]:
        """Return the current rolling game score for a pitcher, or None if unseen."""
        if pitcher_id is None:
            return None
        hist = self._history.get(pitcher_id)
        if not hist:
            return None
        decay = self.params.pitcher_recency_decay
        # Most recent start is at the right end (appended last) -> highest weight.
        weights = [decay ** (len(hist) - 1 - i) for i in range(len(hist))]
        total_w = sum(weights)
        return sum(w * s for w, s in zip(weights, hist)) / total_w

    def adjustment(self, pitcher_id: Optional[int]) -> float:
        """Elo adjustment from a starter's rolling score (0 if no history)."""
        rgs = self.rolling_score(pitcher_id)
        if rgs is None:
            return 0.0
        raw = self.params.pitcher_weight * (rgs - self.params.pitcher_baseline)
        cap = self.params.pitcher_adj_cap
        return max(-cap, min(cap, raw))

    def record(self, pitcher_id: Optional[int], line: Optional[PitcherLine]) -> None:
        """Append a start's game score to the pitcher's rolling history."""
        if pitcher_id is None or line is None:
            return
        window = self.params.pitcher_rolling_window
        hist = self._history.setdefault(pitcher_id, deque(maxlen=window))
        hist.append(game_score(line))
