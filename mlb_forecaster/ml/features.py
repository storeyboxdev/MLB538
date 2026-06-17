"""Assemble the ML feature matrix from engine snapshots + context.

Every feature is strictly *pre-game*. Recent-form features are built from games
strictly before the current one (a shifted rolling window), so the matrix can be
used for honest walk-forward training without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

# Metadata columns carried alongside features (not fed to the model directly).
META_COLS = ["game_pk", "date", "season", "home_team", "away_team", "playoff"]
LABEL_COL = "home_win"          # binary classification target
MARGIN_LABEL_COL = "run_margin"  # regression target: home_score - away_score


def build_team_form(games: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per (game_pk, side) pre-game rolling run differential and win pct.

    Returns a DataFrame indexed by game_pk with columns
    ``home_form_rundiff, away_form_rundiff, home_form_winpct, away_form_winpct``.
    """
    finals = games[games["status"] == "Final"].copy()
    finals["date"] = pd.to_datetime(finals["date"])

    records = []
    for _, r in finals.iterrows():
        records.append((r["home_team"], r["date"], r["game_pk"],
                        r["home_score"] - r["away_score"],
                        int(r["home_score"] > r["away_score"])))
        records.append((r["away_team"], r["date"], r["game_pk"],
                        r["away_score"] - r["home_score"],
                        int(r["away_score"] > r["home_score"])))
    long = pd.DataFrame(records, columns=["team", "date", "game_pk", "rundiff", "win"])
    long = long.sort_values(["team", "date", "game_pk"]).reset_index(drop=True)

    # Shift(1) so the current game is excluded from its own rolling window.
    grp = long.groupby("team", group_keys=False)
    long["form_rundiff"] = grp["rundiff"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["form_winpct"] = grp["win"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    # Re-attach which side each row was, to pivot home/away.
    home_map = finals.set_index("game_pk")["home_team"].to_dict()
    long["side"] = np.where(long.apply(lambda x: home_map.get(x["game_pk"]) == x["team"], axis=1),
                            "home", "away")
    pivot = long.pivot_table(index="game_pk", columns="side",
                             values=["form_rundiff", "form_winpct"], dropna=False)
    pivot.columns = [f"{side}_{val}" for val, side in pivot.columns]
    return pivot


def build_features(engine_df: pd.DataFrame, games: pd.DataFrame,
                   config: Config, *, finals_only: bool = True) -> pd.DataFrame:
    """Build the feature matrix (one row per game) from an engine run.

    With ``finals_only`` the result is the training set (labeled rows only).
    """
    inc = config.raw["features"]["include"]
    window = config.raw["features"].get("recent_form_window", 10)

    df = engine_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    feats = pd.DataFrame(index=df.index)

    if inc.get("elo_diff", True):
        feats["elo_diff"] = df["elo1_pre"] - df["elo2_pre"]
        feats["rating_diff"] = df["rating1_pre"] - df["rating2_pre"]
    if inc.get("team_ratings", True):
        feats["elo1_pre"] = df["elo1_pre"]
        feats["elo2_pre"] = df["elo2_pre"]
    if inc.get("pitcher_adj", True):
        feats["pitcher1_adj"] = df["pitcher1_adj"]
        feats["pitcher2_adj"] = df["pitcher2_adj"]
        feats["pitcher_adj_diff"] = df["pitcher1_adj"] - df["pitcher2_adj"]
    if inc.get("pitcher_rgs", True):
        base = config.raw["elo"].get("pitcher_baseline", 50.0)
        feats["pitcher1_rgs"] = df["pitcher1_rgs"].fillna(base)
        feats["pitcher2_rgs"] = df["pitcher2_rgs"].fillna(base)
    if inc.get("rest_days", True):
        # Cap rest and fill first-game NaNs with a neutral value.
        feats["home_rest"] = df["home_rest"].clip(upper=7).fillna(2)
        feats["away_rest"] = df["away_rest"].clip(upper=7).fillna(2)
        feats["rest_diff"] = feats["home_rest"] - feats["away_rest"]
    if inc.get("month", True):
        feats["month"] = df["date"].dt.month
    if inc.get("playoff", True):
        feats["playoff"] = df["playoff"].astype(int)

    if inc.get("recent_form", True):
        form = build_team_form(games, window)
        form = df[["game_pk"]].merge(form, left_on="game_pk", right_index=True, how="left")
        for col in ["home_form_rundiff", "away_form_rundiff",
                    "home_form_winpct", "away_form_winpct"]:
            feats[col] = form[col].fillna(0.0).to_numpy()

    # Attach metadata + labels (binary win and signed run margin).
    for c in META_COLS:
        feats[c] = df[c].to_numpy()
    feats[LABEL_COL] = df["home_win"].to_numpy()
    feats[MARGIN_LABEL_COL] = (df["home_score"] - df["away_score"]).to_numpy()

    if finals_only:
        feats = feats[feats[LABEL_COL].notna()].reset_index(drop=True)
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Model input columns = everything except metadata and the labels."""
    excluded = set(META_COLS) | {LABEL_COL, MARGIN_LABEL_COL}
    return [c for c in feats.columns if c not in excluded]
