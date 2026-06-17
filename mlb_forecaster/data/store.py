"""Read/write helpers for processed game logs and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def games_path(processed_dir: Path, season: int) -> Path:
    return processed_dir / f"games_{season}.csv"


def write_games(df: pd.DataFrame, processed_dir: Path, season: int) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = games_path(processed_dir, season)
    df.to_csv(path, index=False)
    return path


def read_games(processed_dir: Path, season: int) -> pd.DataFrame:
    path = games_path(processed_dir, season)
    if not path.exists():
        raise FileNotFoundError(f"No processed games for season {season}: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def read_many_seasons(processed_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames = [read_games(processed_dir, s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
