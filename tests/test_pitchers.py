"""Tests for the Bill James game score and rolling pitcher tracker."""

from mlb_forecaster.data.models import PitcherLine
from mlb_forecaster.elo.params import EloParams
from mlb_forecaster.elo.pitchers import PitcherTracker, game_score


def test_game_score_known_line():
    # 7 IP (21 outs), 3 H, 1 ER, 0 unearned, 2 BB, 8 K, 1 HR.
    # 50 + 21 outs + 2*(7-4) innings + 8 K - 2*3 H - 4*1 ER - 2 BB = 73.
    line = PitcherLine(outs=21, hits=3, runs=1, earned_runs=1, walks=2,
                       strikeouts=8, home_runs=1)
    assert game_score(line) == 73


def test_game_score_dominant_higher_than_poor():
    great = PitcherLine(outs=27, hits=1, runs=0, earned_runs=0, walks=0,
                        strikeouts=12, home_runs=0)
    poor = PitcherLine(outs=9, hits=8, runs=7, earned_runs=7, walks=4,
                       strikeouts=1, home_runs=2)
    assert game_score(great) > game_score(poor)


def test_tracker_unseen_pitcher_zero_adjustment():
    tracker = PitcherTracker(EloParams())
    assert tracker.rolling_score(123) is None
    assert tracker.adjustment(123) == 0.0


def test_tracker_recency_weighting():
    params = EloParams(pitcher_recency_decay=0.5, pitcher_rolling_window=10)
    tracker = PitcherTracker(params)
    # First a poor start, then a great one; recency weights the great one more.
    tracker.record(7, PitcherLine(outs=9, hits=8, runs=7, earned_runs=7,
                                  walks=4, strikeouts=1, home_runs=2))
    tracker.record(7, PitcherLine(outs=27, hits=1, runs=0, earned_runs=0,
                                  walks=0, strikeouts=12, home_runs=0))
    rolling = tracker.rolling_score(7)
    simple_avg = (game_score(PitcherLine(outs=9, hits=8, runs=7, earned_runs=7,
                                         walks=4, strikeouts=1, home_runs=2))
                  + game_score(PitcherLine(outs=27, hits=1, runs=0, earned_runs=0,
                                           walks=0, strikeouts=12, home_runs=0))) / 2
    # Recency-weighted score should exceed the simple mean (great start is recent).
    assert rolling > simple_avg


def test_adjustment_capped():
    params = EloParams(pitcher_weight=10.0, pitcher_adj_cap=50.0)
    tracker = PitcherTracker(params)
    tracker.record(9, PitcherLine(outs=27, hits=0, runs=0, earned_runs=0,
                                  walks=0, strikeouts=15, home_runs=0))
    assert abs(tracker.adjustment(9)) <= 50.0
