"""Elo hyperparameters, shared across the engine and the fitter."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import Config


@dataclass
class EloParams:
    mean_rating: float = 1500.0
    k_factor: float = 4.0
    home_field_adv: float = 24.0
    mov_alpha: float = 2.2
    mov_autocorr: float = 0.001
    pitcher_weight: float = 1.0
    pitcher_rolling_window: int = 10
    pitcher_recency_decay: float = 0.85
    pitcher_adj_cap: float = 50.0
    pitcher_baseline: float = 50.0  # league-average Bill James game score
    preseason_reversion: float = 0.33

    @classmethod
    def from_config(cls, config: Config) -> "EloParams":
        e = config.raw["elo"]
        return cls(
            mean_rating=e.get("mean_rating", 1500.0),
            k_factor=e.get("k_factor", 4.0),
            home_field_adv=e.get("home_field_adv", 24.0),
            mov_alpha=e.get("mov_alpha", 2.2),
            mov_autocorr=e.get("mov_autocorr", 0.001),
            pitcher_weight=e.get("pitcher_weight", 1.0),
            pitcher_rolling_window=e.get("pitcher_rolling_window", 10),
            pitcher_recency_decay=e.get("pitcher_recency_decay", 0.85),
            pitcher_adj_cap=e.get("pitcher_adj_cap", 50.0),
            pitcher_baseline=e.get("pitcher_baseline", 50.0),
            preseason_reversion=e.get("preseason_reversion", 0.33),
        )

    def to_dict(self) -> dict:
        return asdict(self)
