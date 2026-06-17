"""Playoff seeding and bracket simulation (current 12-team MLB format).

Per league: 3 division winners (seeds 1-3 by record) + 3 wild cards (seeds 4-6).
- Wild Card Series (Bo3): (3 v 6) and (4 v 5); seeds 1-2 get a bye.
- Division Series (Bo5): 1 v winner(4/5), 2 v winner(3/6).
- League Championship Series (Bo7) -> pennant.
- World Series (Bo7) -> champion.

Series outcomes are sampled from the analytic best-of-N win probability given a
per-game probability derived from team Elo, which is fast and exact in expectation.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..elo.team import expected_score


def series_win_prob(p_game: float, wins_needed: int) -> float:
    """Probability of winning a best-of-(2*wins_needed-1) series at per-game ``p``."""
    p = min(max(p_game, 1e-6), 1 - 1e-6)
    q = 1 - p
    total = 0.0
    for losses in range(wins_needed):  # opponent can lose 0..wins_needed-1 games
        total += math.comb(wins_needed - 1 + losses, losses) * (p ** wins_needed) * (q ** losses)
    return total


def _sim_series(rng: np.random.Generator, a: str, b: str, ratings: dict[str, float],
                wins_needed: int) -> str:
    """Return the winner of a series between a and b (neutral-site Elo)."""
    p = expected_score(ratings.get(a, 1500.0), ratings.get(b, 1500.0), 0.0)
    return a if rng.random() < series_win_prob(p, wins_needed) else b


def order_seeds(team_wins: dict[str, int], rng: np.random.Generator,
                team_meta: dict[str, dict]) -> dict[str, list[str]]:
    """Return ``{league: [seed1..seed6]}`` from a single sim's win totals."""
    out: dict[str, list[str]] = {}
    for league in ("AL", "NL"):
        teams = [t for t in team_wins if team_meta.get(t, {}).get("league") == league]
        # Random jitter breaks ties uniformly (proxy for real tiebreakers).
        def keyf(t: str):
            return (team_wins[t], rng.random())
        # Division winners: best record in each division.
        divisions: dict[str, list[str]] = {}
        for t in teams:
            divisions.setdefault(team_meta[t]["division"], []).append(t)
        div_winners = [max(members, key=keyf) for members in divisions.values()]
        div_winners.sort(key=keyf, reverse=True)
        remaining = [t for t in teams if t not in div_winners]
        remaining.sort(key=keyf, reverse=True)
        wild_cards = remaining[:3]
        out[league] = div_winners[:3] + wild_cards
    return out


def simulate_bracket(seeds: dict[str, list[str]], ratings: dict[str, float],
                     rng: np.random.Generator) -> dict[str, Optional[str]]:
    """Simulate both leagues' brackets + World Series for one sim.

    Returns champions/pennant winners and the set of playoff teams.
    """
    pennant_winners: dict[str, str] = {}
    for league, s in seeds.items():
        if len(s) < 6:
            # Degenerate (shouldn't happen with full leagues); skip cleanly.
            pennant_winners[league] = s[0] if s else None
            continue
        s1, s2, s3, s4, s5, s6 = s[:6]
        # Wild Card round (Bo3).
        w_36 = _sim_series(rng, s3, s6, ratings, 2)
        w_45 = _sim_series(rng, s4, s5, ratings, 2)
        # Division Series (Bo5): 1 vs winner(4/5), 2 vs winner(3/6).
        ds1 = _sim_series(rng, s1, w_45, ratings, 3)
        ds2 = _sim_series(rng, s2, w_36, ratings, 3)
        # LCS (Bo7) -> pennant.
        pennant_winners[league] = _sim_series(rng, ds1, ds2, ratings, 4)
    # World Series (Bo7).
    champ = None
    if pennant_winners.get("AL") and pennant_winners.get("NL"):
        champ = _sim_series(rng, pennant_winners["AL"], pennant_winners["NL"], ratings, 4)
    return {"AL_pennant": pennant_winners.get("AL"),
            "NL_pennant": pennant_winners.get("NL"),
            "champion": champ}
