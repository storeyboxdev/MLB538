"""Tests for the margin-of-victory multiplier."""

from mlb_forecaster.elo.mov import mov_multiplier


def test_larger_margin_larger_multiplier():
    small = mov_multiplier(1, elo_diff_winner=0, alpha=2.2, autocorr=0.001)
    big = mov_multiplier(10, elo_diff_winner=0, alpha=2.2, autocorr=0.001)
    assert big > small


def test_autocorrelation_damping():
    # Favorite winning (positive winner edge) -> damped vs underdog winning.
    favorite = mov_multiplier(5, elo_diff_winner=200, alpha=2.2, autocorr=0.001)
    underdog = mov_multiplier(5, elo_diff_winner=-200, alpha=2.2, autocorr=0.001)
    assert underdog > favorite


def test_positive_and_finite():
    for margin in (1, 3, 7, 15):
        for diff in (-300, 0, 300):
            m = mov_multiplier(margin, diff, 2.2, 0.001)
            assert m > 0 and m < 100


def test_margin_floor():
    # Margin below 1 is floored to 1 (no zero/negative log).
    assert mov_multiplier(0, 0, 2.2, 0.001) == mov_multiplier(1, 0, 2.2, 0.001)
