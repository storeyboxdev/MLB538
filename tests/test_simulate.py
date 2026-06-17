"""Tests for playoff primitives and simulation determinism."""

import math

import pandas as pd

from mlb_forecaster.config import load_config
from mlb_forecaster.forecast.playoffs import series_win_prob
from mlb_forecaster.forecast.simulate import simulate_season


def test_series_win_prob_fair_coin():
    for wins_needed in (1, 2, 3, 4):
        assert math.isclose(series_win_prob(0.5, wins_needed), 0.5, abs_tol=1e-9)


def test_series_win_prob_monotonic():
    assert series_win_prob(0.7, 4) > series_win_prob(0.5, 4) > series_win_prob(0.3, 4)


def _synthetic_league():
    """Build 30 teams (2 leagues x 3 divisions x 5 teams) with games + ratings."""
    leagues = {"AL": ["E", "C", "W"], "NL": ["E", "C", "W"]}
    teams, team_meta = [], {}
    for lg, divs in leagues.items():
        for dv in divs:
            for k in range(5):
                t = f"{lg}{dv}{k}"
                teams.append(t)
                team_meta[t] = {"abbr": t, "league": lg,
                                "division": f"{lg} {dv}"}
    # Completed games: pair teams so higher index beats lower (clear ordering).
    rows, gpk = [], 0
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if i >= j:
                continue
            gpk += 1
            hs, as_ = (5, 2) if i > j else (2, 5)
            rows.append({"game_pk": gpk, "date": "2024-06-01", "season": 2024,
                         "game_type": "R", "home_team": h, "away_team": a,
                         "home_score": hs, "away_score": as_, "status": "Final"})
    # A few scheduled (future) games.
    for k in range(20):
        gpk += 1
        rows.append({"game_pk": gpk, "date": "2024-09-01", "season": 2024,
                     "game_type": "R", "home_team": teams[k % 30],
                     "away_team": teams[(k + 7) % 30], "home_score": None,
                     "away_score": None, "status": "Scheduled"})
    games = pd.DataFrame(rows)
    ratings = {t: 1500.0 + i for i, t in enumerate(teams)}
    return games, ratings, team_meta


def test_simulation_deterministic_with_seed():
    cfg = load_config()
    games, ratings, meta = _synthetic_league()
    a = simulate_season(games, ratings, meta, cfg, n_sims=200, seed=7)
    b = simulate_season(games, ratings, meta, cfg, n_sims=200, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_simulation_structural_invariants():
    cfg = load_config()
    games, ratings, meta = _synthetic_league()
    odds = simulate_season(games, ratings, meta, cfg, n_sims=300, seed=1)
    # World Series probabilities sum to 1 across all teams.
    assert math.isclose(odds["win_ws"].sum(), 1.0, abs_tol=1e-6)
    # Each sim seats 6 playoff teams per league (12-team format), so the
    # make-playoffs probabilities sum to 6 within each league.
    for lg in ("AL", "NL"):
        assert math.isclose(odds[odds["league"] == lg]["make_playoffs"].sum(),
                            6.0, abs_tol=1e-6)
        assert math.isclose(odds[odds["league"] == lg]["win_pennant"].sum(),
                            1.0, abs_tol=1e-6)
