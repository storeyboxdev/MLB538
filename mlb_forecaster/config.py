"""Configuration loading.

The whole project is driven by ``config.yaml`` at the repo root. We load it once
into a nested dict wrapped in :class:`Config`, which provides attribute-style and
dict-style access plus resolved, absolute data paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (mlb_forecaster/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class Config:
    """Lightweight wrapper over the parsed YAML config."""

    def __init__(self, data: dict[str, Any], root: Path = REPO_ROOT):
        self._data = data
        self.root = root

    # --- access helpers -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    # --- resolved paths -------------------------------------------------
    def _path(self, key: str) -> Path:
        p = Path(self._data["data"][key])
        if not p.is_absolute():
            p = self.root / p
        return p

    @property
    def raw_dir(self) -> Path:
        return self._path("raw_dir")

    @property
    def processed_dir(self) -> Path:
        return self._path("processed_dir")

    @property
    def features_dir(self) -> Path:
        return self._path("features_dir")

    @property
    def models_dir(self) -> Path:
        return self._path("models_dir")

    @property
    def output_dir(self) -> Path:
        return self._path("output_dir")

    def ensure_dirs(self) -> None:
        """Create all data directories if missing."""
        for key in ("raw_dir", "processed_dir", "features_dir", "models_dir", "output_dir"):
            self._path(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration from ``path`` (defaults to repo-root config.yaml)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, root=cfg_path.resolve().parent)
