"""Monte Carlo simulation of the remaining regular season + playoffs.

Regular-season game probabilities come from the trained model (or Elo fallback)
using current ratings and neutral context. Each simulated season produces final
win totals, from which playoff seeds and a bracket are drawn.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..elo.team import expected_score
from .playoffs import order_seeds, simulate_bracket

# Feature names the model may consume; we build all and let the predictor subset.
_BASELINE_FORM = 0.0


def current_standings(games_season: pd.DataFrame) -> dict[str, list[int]]:
    """Return ``{team: [wins, losses]}`` from completed regular-season games."""
    reg = games_season[(games_season["game_type"] == "R") &
                        (games_season["status"] == "Final") &
                        games_season["home_score"].notna() &
                        games_season["away_score"].notna()]
    rec: dict[str, list[int]] = {}
    for _, g in reg.iterrows():
        h, a = g["home_team"], g["away_team"]
        rec.setdefault(h, [0, 0])
        rec.setdefault(a, [0, 0])
        if g["home_score"] > g["away_score"]:
            rec[h][0] += 1; rec[a][1] += 1
        else:
            rec[a][0] += 1; rec[h][1] += 1
    return rec


def current_run_diff(games_season: pd.DataFrame) -> dict[str, float]:
    """Return ``{team: run differential}`` from completed regular-season games."""
    reg = games_season[(games_season["game_type"] == "R") &
                        (games_season["status"] == "Final") &
                        games_season["home_score"].notna() &
                        games_season["away_score"].notna()]
    diff: dict[str, float] = {}
    for _, g in reg.iterrows():
        m = g["home_score"] - g["away_score"]
        diff[g["home_team"]] = diff.get(g["home_team"], 0.0) + m
        diff[g["away_team"]] = diff.get(g["away_team"], 0.0) - m
    return diff


def remaining_games(games_season: pd.DataFrame) -> pd.DataFrame:
    """Scheduled (not yet final) regular-season games."""
    return games_season[(games_season["game_type"] == "R") &
                         (games_season["status"] != "Final")].copy()


def build_matchup_features(rem: pd.DataFrame, ratings: dict[str, float],
                           baseline: float) -> pd.DataFrame:
    """Construct a model feature matrix for future games (neutral context)."""
    rem = rem.copy()
    rem["date"] = pd.to_datetime(rem["date"])
    e1 = rem["home_team"].map(lambda t: ratings.get(t, 1500.0)).to_numpy()
    e2 = rem["away_team"].map(lambda t: ratings.get(t, 1500.0)).to_numpy()
    n = len(rem)
    feats = pd.DataFrame({
        "elo_diff": e1 - e2,
        "rating_diff": e1 - e2,
        "elo1_pre": e1,
        "elo2_pre": e2,
        "pitcher1_adj": np.zeros(n),
        "pitcher2_adj": np.zeros(n),
        "pitcher_adj_diff": np.zeros(n),
        "pitcher1_rgs": np.full(n, baseline),
        "pitcher2_rgs": np.full(n, baseline),
        "home_rest": np.full(n, 2.0),
        "away_rest": np.full(n, 2.0),
        "rest_diff": np.zeros(n),
        "month": rem["date"].dt.month.to_numpy(),
        "playoff": np.zeros(n, dtype=int),
        "home_form_rundiff": np.full(n, _BASELINE_FORM),
        "away_form_rundiff": np.full(n, _BASELINE_FORM),
        "home_form_winpct": np.full(n, _BASELINE_FORM),
        "away_form_winpct": np.full(n, _BASELINE_FORM),
    })
    return feats


def simulate_season(games_season: pd.DataFrame, ratings: dict[str, float],
                    team_meta: dict[str, dict], config: Config,
                    prob_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
                    n_sims: Optional[int] = None,
                    seed: Optional[int] = None,
                    margin_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None
                    ) -> pd.DataFrame:
    """Run the Monte Carlo and return a per-team odds DataFrame.

    ``prob_fn`` maps a matchup feature matrix -> P(home win). If None, an Elo
    fallback (with HFA) is used. ``margin_fn`` (optional) maps the same matrix to
    an expected home run margin, used to project end-of-season run differential.
    """
    fcfg = config.raw["forecast"]
    n_sims = n_sims or fcfg.get("n_sims", 10000)
    seed = seed if seed is not None else fcfg.get("random_seed", 538)
    hfa = config.raw["elo"].get("home_field_adv", 24.0)
    baseline = config.raw["elo"].get("pitcher_baseline", 50.0)
    rng = np.random.default_rng(seed)

    # Universe of teams = those in the metadata that appear this season.
    standings = current_standings(games_season)
    teams = sorted([t for t in team_meta if t in standings] or list(standings))
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    wins0 = np.array([standings.get(t, [0, 0])[0] for t in teams], dtype=np.int32)

    # Projected run differential starts from games already played.
    rundiff0 = current_run_diff(games_season)
    proj_run_diff = np.array([rundiff0.get(t, 0.0) for t in teams], dtype=float)

    rem = remaining_games(games_season)
    if rem.empty:
        p_home = np.array([])
        home_idx = np.array([], dtype=int)
        away_idx = np.array([], dtype=int)
    else:
        feats = build_matchup_features(rem, ratings, baseline)
        if prob_fn is not None:
            p_home = np.asarray(prob_fn(feats), dtype=float)
        else:
            p_home = expected_score(feats["elo1_pre"].to_numpy(),
                                    feats["elo2_pre"].to_numpy(), hfa)
        home_idx = rem["home_team"].map(idx).to_numpy()
        away_idx = rem["away_team"].map(idx).to_numpy()
        if margin_fn is not None:
            exp_margin = np.asarray(margin_fn(feats), dtype=float)
            np.add.at(proj_run_diff, home_idx, exp_margin)
            np.add.at(proj_run_diff, away_idx, -exp_margin)

    # Vectorized regular-season win totals: wins[n_sims, n_teams].
    wins = np.tile(wins0, (n_sims, 1)).astype(np.int32)
    for g in range(len(p_home)):
        hw = rng.random(n_sims) < p_home[g]
        wins[hw, home_idx[g]] += 1
        wins[~hw, away_idx[g]] += 1

    # Tallies.
    made = np.zeros(n_teams, dtype=np.int64)
    div = np.zeros(n_teams, dtype=np.int64)
    pennant = np.zeros(n_teams, dtype=np.int64)
    champ = np.zeros(n_teams, dtype=np.int64)

    for s in range(n_sims):
        team_wins = {teams[i]: int(wins[s, i]) for i in range(n_teams)}
        seeds = order_seeds(team_wins, rng, team_meta)
        for league, slist in seeds.items():
            for t in slist:
                made[idx[t]] += 1
            for t in slist[:3]:
                div[idx[t]] += 1
        res = simulate_bracket(seeds, ratings, rng)
        for key in ("AL_pennant", "NL_pennant"):
            t = res.get(key)
            if t is not None:
                pennant[idx[t]] += 1
        if res.get("champion") is not None:
            champ[idx[res["champion"]]] += 1

    proj_wins = wins.mean(axis=0)
    df = pd.DataFrame({
        "team": teams,
        "league": [team_meta.get(t, {}).get("league", "") for t in teams],
        "division": [team_meta.get(t, {}).get("division", "") for t in teams],
        "current_wins": wins0,
        "proj_wins": np.round(proj_wins, 1),
        "proj_run_diff": np.round(proj_run_diff, 0),
        "make_playoffs": made / n_sims,
        "win_division": div / n_sims,
        "win_pennant": pennant / n_sims,
        "win_ws": champ / n_sims,
    })
    df = df.sort_values(["league", "win_ws", "make_playoffs"], ascending=[True, False, False])
    return df.reset_index(drop=True)
