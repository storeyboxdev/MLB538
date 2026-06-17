"""Margin-of-victory multiplier with autocorrelation damping.

Follows FiveThirtyEight's approach: bigger margins increase the rating change,
but the effect is damped when a heavy favorite wins (and amplified when an
underdog wins), which corrects for the autocorrelation between rating gaps and
expected margins.
"""

from __future__ import annotations

import math


def mov_multiplier(margin: float, elo_diff_winner: float,
                   alpha: float, autocorr: float) -> float:
    """Compute the MOV multiplier.

    Parameters
    ----------
    margin : absolute run differential of the game (>= 1).
    elo_diff_winner : winner's pre-game Elo minus loser's pre-game Elo
        (positive when the favorite won). Includes any pre-game adjustments
        that fed the prediction (HFA + pitcher), so the damping matches the
        expectation actually used.
    alpha : shape parameter (538 uses ~2.2).
    autocorr : autocorrelation damping coefficient (538 uses ~0.001).
    """
    margin = max(abs(margin), 1.0)
    denom = elo_diff_winner * autocorr + alpha
    # Guard against degenerate/negative denominators from extreme inputs.
    if denom <= 1e-6:
        denom = 1e-6
    return math.log(margin + 1.0) * (alpha / denom)
