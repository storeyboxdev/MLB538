"""Between-season reversion of team ratings toward the mean."""

from __future__ import annotations


def revert_to_mean(rating: float, mean_rating: float, fraction: float) -> float:
    """Pull ``rating`` ``fraction`` of the way toward ``mean_rating``.

    fraction=0 keeps the carryover rating, fraction=1 fully resets to the mean.
    """
    return rating + fraction * (mean_rating - rating)
