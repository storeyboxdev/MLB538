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

## Results

Walk-forward backtest over 8 seasons (2017–2025, excluding the COVID-shortened
2020), evaluated only on seasons the model never trained on (~14,800 out-of-sample
games). Lower is better; a coin flip scores **0.6931** log loss.

| Model | Log loss | Brier | Margin MAE |
|---|---|---|---|
| Elo (team only) | 0.6765 | 0.2418 | — |
| **Elo + pitcher adjustment** | **0.6754** | **0.2413** | — |
| Logistic regression | 0.6768 | 0.2419 | — |
| LightGBM | 0.6841 | 0.2455 | — |
| Run-margin (Ridge) | 0.6771 | 0.2421 | 3.42 runs |

Takeaways:

- **Calibration is excellent** — predicted probabilities match observed results
  across every bucket.
- **It reproduces reality** — for 2024 it gave the Dodgers the top World Series
  odds (they won) and its simulated playoff field matched the actual one.
- **Pitcher data adds real, measurable lift** — given starting-pitcher box scores,
  the hyperparameter fitter raised the pitcher weight from 1.0 to ~1.6 on its own,
  and pitcher-adjusted Elo beats team-only Elo across all test games.
- **Baseball is hard** — MLB is the least predictable major sport at the
  single-game level; a finely-tuned, pitcher-aware Elo is near the ceiling, and the
  ML layer matches it while staying calibrated. The framework's payoff is that it
  will automatically exploit richer features (bullpen, Statcast, park, weather) as
  they're added.

Regenerate these numbers with `mlbfc backtest`.

## Tests

```bash
pytest
```

28 tests cover the Elo math, MOV multiplier, pitcher game score, the run-margin
model, a feature-leakage guard, and simulation determinism.

## License

MIT — see [LICENSE](LICENSE).
