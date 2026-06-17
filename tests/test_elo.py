"""Tests for core Elo math and the engine."""

import math

import pandas as pd

from mlb_forecaster.elo.params import EloParams
from mlb_forecaster.elo.preseason import revert_to_mean
from mlb_forecaster.elo.team import expected_score, update_rating
from mlb_forecaster.elo.engine import run_engine


def test_expected_score_equal_ratings_no_hfa():
    assert expected_score(1500, 1500, 0.0) == 0.5


def test_expected_score_home_field_advantage():
    # 24 Elo points of HFA between equal teams -> home favored ~53.4%.
    p = expected_score(1500, 1500, 24.0)
    assert 0.53 < p < 0.54


def test_expected_score_monotonic_in_rating_gap():
    assert expected_score(1600, 1500, 0) > expected_score(1550, 1500, 0) > 0.5


def test_expected_score_symmetry():
    p = expected_score(1600, 1500, 0)
    q = expected_score(1500, 1600, 0)
    assert math.isclose(p + q, 1.0, abs_tol=1e-9)


def test_update_rating_winner_gains_loser_loses():
    # Even teams, home wins (actual=1, expected=0.5): rating should increase.
    new = update_rating(1500, k=4, mov_mult=1.0, actual=1.0, expected=0.5)
    assert new > 1500
    # Zero-sum: loser's symmetric update mirrors the gain.
    loser = update_rating(1500, k=4, mov_mult=1.0, actual=0.0, expected=0.5)
    assert math.isclose((new - 1500), (1500 - loser), abs_tol=1e-9)


def test_revert_to_mean():
    assert revert_to_mean(1600, 1500, 0.0) == 1600
    assert revert_to_mean(1600, 1500, 1.0) == 1500
    assert revert_to_mean(1600, 1500, 0.5) == 1550


def _toy_games():
    # Two games: A beats B at home, then B beats A at home.
    return pd.DataFrame([
        {"game_pk": 1, "date": "2023-04-01", "season": 2023, "game_type": "R",
         "home_team": "A", "away_team": "B", "home_score": 5, "away_score": 2,
         "status": "Final", "playoff": False,
         "home_pitcher_id": None, "away_pitcher_id": None,
         "home_starter_id": None, "away_starter_id": None},
        {"game_pk": 2, "date": "2023-04-03", "season": 2023, "game_type": "R",
         "home_team": "B", "away_team": "A", "home_score": 4, "away_score": 1,
         "status": "Final", "playoff": False,
         "home_pitcher_id": None, "away_pitcher_id": None,
         "home_starter_id": None, "away_starter_id": None},
    ])


def test_engine_runs_and_updates():
    out = run_engine(_toy_games(), EloParams())
    assert len(out) == 2
    # Winner of game 1 (home A) should end above the mean after that game.
    g1 = out[out["game_pk"] == 1].iloc[0]
    assert g1["elo1_post"] > 1500 > g1["elo2_post"]
    # Probabilities are valid.
    assert 0 < g1["rating_prob1"] < 1


def test_engine_rest_days():
    out = run_engine(_toy_games(), EloParams())
    g2 = out[out["game_pk"] == 2].iloc[0]
    # Both teams had a game on 4/1, next on 4/3 -> 2 days rest.
    assert g2["home_rest"] == 2
    assert g2["away_rest"] == 2
