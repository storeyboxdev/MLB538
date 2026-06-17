"""Core Elo math: expected score and rating update."""

from __future__ import annotations


def expected_score(elo_home: float, elo_away: float, home_field_adv: float) -> float:
    """Home team's expected score (win probability) under the logistic Elo curve.

    ``P(home) = 1 / (1 + 10^(-(elo_home - elo_away + HFA) / 400))``
    """
    diff = (elo_home - elo_away + home_field_adv) / 400.0
    return 1.0 / (1.0 + 10.0 ** (-diff))


def update_rating(rating: float, k: float, mov_mult: float,
                  actual: float, expected: float) -> float:
    """Return the updated rating after one game.

    ``actual`` is 1.0 for a win, 0.0 for a loss (from this team's perspective).
    The margin-of-victory multiplier scales the standard Elo delta.
    """
    return rating + k * mov_mult * (actual - expected)
