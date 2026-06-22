"""Compare model/training variants side-by-side under walk-forward CV.

This is the engine behind ``mlbfc compare``. Each *preset* (defined in
``experiments.yaml``) is a sparse override applied on top of ``config.yaml`` —
it may tweak the ML hyperparameters, the calibration method, and/or the feature
toggles. Every variant is scored with the SAME expanding-window cross-validation
used in training (:func:`mlb_forecaster.ml.train.walk_forward_cv`), so the
numbers are directly comparable to what ``train``/``backtest`` report.

Scope note: the Elo engine is held FIXED across variants (run once). Tinkering
with Elo hyperparameters is intentionally out of scope here — change those in
``config.yaml`` and re-run ``fit-elo``/``backtest`` instead.

Nothing is persisted as a model: this is a report-only tool. To adopt a winner,
copy its settings into ``config.yaml`` and run ``mlbfc train``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from .config import Config
from .elo.engine import run_engine
from .elo.params import EloParams
from .ml.features import build_features, feature_columns
from .ml.train import walk_forward_cv

Logger = Callable[[str], None]

DEFAULT_KINDS: tuple[str, ...] = ("logistic", "lightgbm")


def load_presets(path: str | Path) -> dict[str, dict]:
    """Read ``experiments.yaml`` and return its ``presets`` mapping."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    presets = data.get("presets")
    if not presets:
        raise ValueError(f"No 'presets:' section found in {path}")
    return presets


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a deep-merged copy of ``base`` with ``override`` taking precedence."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _feature_signature(merged_raw: dict) -> str:
    """Stable key for a variant's feature config, so identical ones reuse a build."""
    return json.dumps(merged_raw.get("features", {}), sort_keys=True, default=str)


def run_comparison(config: Config, games: pd.DataFrame, presets: dict[str, dict],
                   names: list[str] | None = None,
                   log: Logger = print) -> dict[str, Any]:
    """Score each preset under walk-forward CV; return a ranked, JSON-safe report.

    Returns ``{"seasons": [...], "results": [...ranked...], "best": name}`` where
    each result has the selected family's ``log_loss``/``brier``/``n`` plus a
    per-family breakdown.
    """
    if names:
        missing = [n for n in names if n not in presets]
        if missing:
            raise ValueError(f"Unknown preset(s): {', '.join(missing)}")
        presets = {n: presets[n] for n in names}

    # Elo is fixed across variants: run the engine once and reuse it.
    params = _get_params(config)
    eng = run_engine(games, params)

    # Cache built feature frames keyed by feature-config signature so that
    # ml-only variants share a single (expensive) build_features pass.
    feat_cache: dict[str, pd.DataFrame] = {}

    results = []
    for name, override in presets.items():
        override = override or {}
        if "elo" in override:
            log(f"[compare] note: preset '{name}' sets 'elo:' — ignored "
                f"(Elo is fixed during comparison)")
        kinds = tuple(override.get("kinds", DEFAULT_KINDS))

        variant_raw = _deep_merge(config.raw, override)
        variant_cfg = Config(variant_raw, root=config.root)

        sig = _feature_signature(variant_raw)
        feats = feat_cache.get(sig)
        if feats is None:
            feats = build_features(eng, games, variant_cfg)
            feat_cache[sig] = feats
        cols = feature_columns(feats)
        method = variant_raw["ml"].get("calibration", "sigmoid")

        families = {}
        for kind in kinds:
            rep = walk_forward_cv(kind, variant_cfg, feats, cols, method)
            families[kind] = {k: rep[k] for k in ("log_loss", "brier", "n")}

        best_family = min(families, key=lambda k: families[k]["log_loss"])
        results.append({
            "name": name,
            "family": best_family,
            "calibration": method,
            **families[best_family],
            "families": families,
        })
        log(f"[compare] {name:>20}: {best_family} "
            f"log_loss={families[best_family]['log_loss']:.4f} "
            f"brier={families[best_family]['brier']:.4f}")

    results.sort(key=lambda r: r["log_loss"])
    seasons = sorted(int(s) for s in feats["season"].unique()) if results else []
    return {
        "seasons": seasons,
        "results": results,
        "best": results[0]["name"] if results else None,
    }


def _get_params(config: Config) -> EloParams:
    """Fitted Elo params from the registry if present, else config defaults.

    Imported lazily to avoid a circular import with :mod:`pipeline`.
    """
    from .ml.registry import load_elo_params

    fitted = load_elo_params(config.models_dir, "latest")
    return fitted or EloParams.from_config(config)
