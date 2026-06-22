"""Forward chronological Elo pass producing ratings + pre-game snapshots.

The output mirrors FiveThirtyEight's mlb-elo dataset: a simple team-only Elo
(``elo*``) and a pitcher-adjusted rating (``rating*``), with the pre-game state
captured for every game (the ML training input) and post-game ratings for finals.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from ..data.models import PitcherLine
from .mov import mov_multiplier
from .params import EloParams
from .pitchers import PitcherTracker
from .preseason import revert_to_mean
from .team import expected_score, update_rating


def _pitcher_line_from_row(row, side: str) -> Optional[PitcherLine]:
    """Reconstruct a PitcherLine from flattened row columns, or None if absent."""
    outs = getattr(row, f"{side}_outs", None)
    if outs is None or (isinstance(outs, float) and math.isnan(outs)):
        return None
    def g(field: str) -> int:
        v = getattr(row, f"{side}_{field}", 0)
        return int(v) if v == v else 0  # NaN -> 0
    return PitcherLine(
        outs=int(outs),
        hits=g("hits"),
        runs=g("runs"),
        earned_runs=g("earned_runs"),
        walks=g("walks"),
        strikeouts=g("strikeouts"),
        home_runs=g("home_runs"),
    )


def _val(row, name, default=None):
    v = getattr(row, name, default)
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    return v


def run_engine(games: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    """Run the Elo engine over ``games`` (any number of seasons) chronologically.

    Returns a DataFrame of per-game rows with pre/post elo ratings, probabilities,
    pitcher adjustments, rest, and the home-win label (NaN for non-final games).
    """
    df = games.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    last_played: dict[str, object] = {}
    tracker = PitcherTracker(params)
    prev_season: Optional[int] = None
    out_rows: list[dict] = []

    for row in df.itertuples(index=False):
        season = int(row.season)
        if prev_season is not None and season != prev_season:
            for team in list(ratings):
                ratings[team] = revert_to_mean(
                    ratings[team], params.mean_rating, params.preseason_reversion
                )
        prev_season = season

        home, away = row.home_team, row.away_team
        eh = ratings.setdefault(home, params.mean_rating)
        ea = ratings.setdefault(away, params.mean_rating)

        # Pre-game pitcher adjustments use the *probable* starter (known pre-game).
        home_pid = _val(row, "home_pitcher_id")
        away_pid = _val(row, "away_pitcher_id")
        adj_h = tracker.adjustment(int(home_pid) if home_pid is not None else None)
        adj_a = tracker.adjustment(int(away_pid) if away_pid is not None else None)
        rgs_h = tracker.rolling_score(int(home_pid) if home_pid is not None else None)
        rgs_a = tracker.rolling_score(int(away_pid) if away_pid is not None else None)

        eff_h, eff_a = eh + adj_h, ea + adj_a
        prob_elo = expected_score(eh, ea, params.home_field_adv)
        prob_rating = expected_score(eff_h, eff_a, params.home_field_adv)

        gdate = row.date
        rest_h = (gdate - last_played[home]).days if home in last_played else float("nan")
        rest_a = (gdate - last_played[away]).days if away in last_played else float("nan")

        rec = {
            "game_pk": row.game_pk,
            "date": gdate,
            "season": season,
            "playoff": bool(_val(row, "playoff", False)),
            "home_team": home,
            "away_team": away,
            "elo1_pre": eh,
            "elo2_pre": ea,
            "pitcher1_adj": adj_h,
            "pitcher2_adj": adj_a,
            "pitcher1_rgs": rgs_h if rgs_h is not None else float("nan"),
            "pitcher2_rgs": rgs_a if rgs_a is not None else float("nan"),
            "rating1_pre": eff_h,
            "rating2_pre": eff_a,
            "elo_prob1": prob_elo,
            "rating_prob1": prob_rating,
            "home_rest": rest_h,
            "away_rest": rest_a,
            "home_score": _val(row, "home_score"),
            "away_score": _val(row, "away_score"),
        }

        is_final = (
            str(_val(row, "status", "")) == "Final"
            and rec["home_score"] is not None
            and rec["away_score"] is not None
        )

        if is_final:
            hs, as_ = float(rec["home_score"]), float(rec["away_score"])
            result_h = 1.0 if hs > as_ else 0.0
            margin = abs(hs - as_)
            # Winner's pre-game effective edge (incl. HFA) for autocorrelation damping.
            full_h = eff_h + params.home_field_adv
            full_a = eff_a
            winner_diff = (full_h - full_a) if result_h == 1.0 else (full_a - full_h)
            mov = mov_multiplier(margin, winner_diff, params.mov_alpha, params.mov_autocorr)
            delta = params.k_factor * mov * (result_h - prob_rating)
            ratings[home] = eh + delta
            ratings[away] = ea - delta

            # Record the actual starts into the rolling tracker.
            home_sid = _val(row, "home_starter_id", home_pid)
            away_sid = _val(row, "away_starter_id", away_pid)
            tracker.record(int(home_sid) if home_sid is not None else None,
                           _pitcher_line_from_row(row, "home"))
            tracker.record(int(away_sid) if away_sid is not None else None,
                           _pitcher_line_from_row(row, "away"))
            last_played[home] = gdate
            last_played[away] = gdate

            rec["elo1_post"] = ratings[home]
            rec["elo2_post"] = ratings[away]
            rec["home_win"] = result_h
            rec["mov_mult"] = mov
        else:
            rec["elo1_post"] = eh
            rec["elo2_post"] = ea
            rec["home_win"] = float("nan")
            rec["mov_mult"] = float("nan")

        out_rows.append(rec)

    return pd.DataFrame(out_rows)


def final_ratings(engine_df: pd.DataFrame) -> dict[str, float]:
    """Extract each team's most recent post-game team Elo from an engine run."""
    ratings: dict[str, float] = {}
    for row in engine_df.sort_values(["date", "game_pk"]).itertuples(index=False):
        ratings[row.home_team] = row.elo1_post
        ratings[row.away_team] = row.elo2_post
    return ratings
