# MLB Forecaster

An ML-driven, FiveThirtyEight-style forecaster for Major League Baseball.

Elo ratings (with starting-pitcher adjustments) are used as **features** for a
trained, calibrated prediction model. The model is fit on previous seasons,
validated with walk-forward cross-validation, and self-corrects as new games
arrive. A Monte Carlo simulation of the remaining schedule produces
playoff / division / World Series odds.

## How it works

1. **Ingest** schedules, results, and pitcher box lines from the free official
   MLB Stats API (`statsapi.mlb.com`).
2. **Elo engine** computes team ratings forward in time, with home-field
   advantage, a margin-of-victory multiplier (autocorrelation-damped), and a
   recency-weighted starting-pitcher adjustment. Its hyperparameters are *fit to
   history*, not hardcoded.
3. **ML layer** trains logistic regression and LightGBM on pre-game Elo features
   plus context (rest, travel, park, recent form), with probability calibration.
4. **Forecast** runs a Monte Carlo over the remaining schedule using the trained
   model's win probabilities.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
mlbfc scrape   --seasons 2017-2025      # ingest games
mlbfc rate     --seasons 2017-2025      # run Elo engine -> ratings + feature snapshots
mlbfc fit-elo  --seasons 2017-2024      # optimize Elo hyperparameters
mlbfc train    --through 2024           # train LR + LightGBM, calibrate, save best
mlbfc forecast --season 2025 --sims 10000
mlbfc update   --season 2025            # self-correct: online Elo + periodic retrain
mlbfc backtest --through 2024           # log-loss / Brier / calibration, model comparison
```

All tunables live in `config.yaml`.

## Tests

```bash
pytest
```
