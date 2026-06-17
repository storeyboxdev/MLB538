"""Versioned persistence for the trained model + Elo params + metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

from ..elo.params import EloParams

MODEL_FILE = "model.joblib"
MARGIN_FILE = "margin_model.joblib"
ELO_FILE = "elo_params.json"
META_FILE = "metadata.json"


def _write_bundle(dest: Path, calibrated_model, cols: list[str],
                  elo_params: EloParams, report: dict[str, Any], version: str,
                  margin_model=None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": calibrated_model, "cols": cols}, dest / MODEL_FILE)
    if margin_model is not None:
        joblib.dump({"margin": margin_model, "cols": margin_model.feature_columns},
                    dest / MARGIN_FILE)
    with open(dest / ELO_FILE, "w", encoding="utf-8") as fh:
        json.dump(elo_params.to_dict(), fh, indent=2)
    with open(dest / META_FILE, "w", encoding="utf-8") as fh:
        json.dump({"version": version, "report": report,
                   "created": datetime.now(timezone.utc).isoformat()}, fh, indent=2)


def save_model(models_dir: Path, calibrated_model, cols: list[str],
               elo_params: EloParams, report: dict[str, Any],
               version: Optional[str] = None, margin_model=None) -> Path:
    """Persist a model bundle; also updates the ``latest`` pointer directory."""
    version = version or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = models_dir / version
    _write_bundle(dest, calibrated_model, cols, elo_params, report, version, margin_model)
    _write_bundle(models_dir / "latest", calibrated_model, cols, elo_params,
                  report, version, margin_model)
    return dest


def load_model(models_dir: Path, version: str = "latest") -> dict[str, Any]:
    """Load a model bundle: ``{model, cols, elo_params, metadata, margin}``."""
    src = models_dir / version
    bundle = joblib.load(src / MODEL_FILE)
    with open(src / ELO_FILE, "r", encoding="utf-8") as fh:
        elo_params = EloParams(**json.load(fh))
    meta = {}
    meta_path = src / META_FILE
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    margin = None
    margin_path = src / MARGIN_FILE
    if margin_path.exists():
        margin = joblib.load(margin_path)["margin"]
    return {"model": bundle["model"], "cols": bundle["cols"],
            "elo_params": elo_params, "metadata": meta, "margin": margin}


def load_elo_params(models_dir: Path, version: str = "latest") -> Optional[EloParams]:
    """Load just the fitted Elo params, if a model has been saved."""
    path = models_dir / version / ELO_FILE
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return EloParams(**json.load(fh))
