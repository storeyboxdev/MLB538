"""Tests guarding against feature leakage."""

import pandas as pd

from mlb_forecaster.config import load_config
from mlb_forecaster.elo.engine import run_engine
from mlb_forecaster.elo.params import EloParams
from mlb_forecaster.ml.features import (LABEL_COL, build_features,
                                        build_team_form, feature_columns)

# Columns that encode the outcome or post-game state and must never be features.
FORBIDDEN = {"home_score", "away_score", "elo1_post", "elo2_post",
             "mov_mult", LABEL_COL}


def _games():
    rows = []
    teams = ["A", "B"]
    # A scores escalating run differentials so its rolling form is well-defined.
    schedule = [
        ("2023-04-01", "A", "B", 5, 1),
        ("2023-04-02", "B", "A", 1, 4),
        ("2023-04-03", "A", "B", 7, 0),
    ]
    for i, (d, h, a, hs, as_) in enumerate(schedule):
        rows.append({"game_pk": i + 1, "date": d, "season": 2023, "game_type": "R",
                     "home_team": h, "away_team": a, "home_score": hs, "away_score": as_,
                     "status": "Final", "playoff": False,
                     "home_pitcher_id": None, "away_pitcher_id": None,
                     "home_starter_id": None, "away_starter_id": None})
    return pd.DataFrame(rows)


def test_feature_columns_exclude_outcome():
    cfg = load_config()
    games = _games()
    eng = run_engine(games, EloParams())
    feats = build_features(eng, games, cfg)
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"leaky columns present: {cols & FORBIDDEN}"


def test_form_excludes_current_game():
    # A team's recent-form for its FIRST game must not see that game's result.
    form = build_team_form(_games(), window=10)
    # Game 1 is the first game for both A and B -> no prior games -> NaN form.
    assert pd.isna(form.loc[1, "home_form_rundiff"])
    assert pd.isna(form.loc[1, "away_form_rundiff"])
    # Game 2: away team is A, whose only prior game (game 1) was a +4 run diff.
    assert form.loc[2, "away_form_rundiff"] == 4.0


def test_no_future_information_in_form():
    # For game 2 (home B), B's prior result was game 1: B lost by 4 (-4 diff).
    form = build_team_form(_games(), window=10)
    assert form.loc[2, "home_form_rundiff"] == -4.0
