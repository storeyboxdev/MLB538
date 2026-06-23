# Model Training

This document describes how the MLB forecaster builds its training data, which features
feed the models, and how models are fit, calibrated, and selected.

## Data preprocessing

Training starts from processed game logs (one row per game, scraped from the MLB Stats
API). The pipeline walks games **chronologically** — sorted by date and `game_pk` — and
builds a per-game snapshot of everything known **before first pitch**

### Elo engine processing

As part of the data preprocessing, for each game, the Elo engine:

1. **Pre-game team Elo** — looks up each team's current rating (`elo1_pre` for home,
   `elo2_pre` for away). New teams start at the configured mean (default 1500). Between
   seasons, ratings partially revert toward that mean (preseason reversion).

2. **Starting pitchers** — uses the **probable** starter IDs announced pre-game. A
   `PitcherTracker` maintains a recency-weighted rolling Bill James game score for each
   pitcher (from their recent starts) — a single-number summary of one start from outs,
   strikeouts, walks, hits, and runs. That score is converted into an Elo adjustment
   (`pitcher1_adj`, `pitcher2_adj`) and added to the team rating to produce an
   **effective pre-game rating** (`rating1_pre`, `rating2_pre`).

   > [Bill James game score](https://en.wikipedia.org/wiki/Game_score) is a single
   > number rating a pitcher's performance in one game from his line stats.

3. **Rest days** — counts calendar days since each team's last completed game
   (`home_rest`, `away_rest`).

4. **Post-game Elo** — for **final** games only, ratings are updated using the
   margin-of-victory multiplier and the actual result. Post-game values (`elo1_post`,
   `elo2_post`) are stored for downstream use (e.g. current standings) but are **not**
   used as ML features — they would leak the outcome.

5. **Pitcher history update** — after a final game, the **actual** starter's box-score
   line is recorded into the pitcher tracker for future games.

The engine output (`ratings.csv`) is one row per game with pre-game snapshots, context,
and labels (`home_win`, scores) for completed games.

### Feature matrix

`build_features()` merges the engine output with additional context from the raw game
logs. Every feature is strictly **pre-game** — recent-form windows use a shifted rolling
average so the current game is never included in its own features. Only rows with a
known result (`home_win` not null) are kept for training.

Run the full training pipeline with:

```bash
mlbfc train --through 2024
```

## Features

Feature groups can be toggled in `config.yaml` under `features.include`. By default, the
model sees the following columns (home team is side 1, away is side 2):

| Feature | Description |
|---|---|
| `elo_diff` | Home pre-game team Elo minus away pre-game team Elo |
| `rating_diff` | Home effective pre-game rating minus away (team Elo + pitcher adjustment) |
| `elo1_pre` | Home team pre-game Elo |
| `elo2_pre` | Away team pre-game Elo |
| `pitcher1_adj` | Home starter's Elo adjustment from rolling game score |
| `pitcher2_adj` | Away starter's Elo adjustment |
| `pitcher_adj_diff` | `pitcher1_adj − pitcher2_adj` |
| `pitcher1_rgs` | Home starter's rolling game score (missing → baseline of 50) |
| `pitcher2_rgs` | Away starter's rolling game score |
| `home_rest` | Days since home team's last game (capped at 7; first game → 2) |
| `away_rest` | Days since away team's last game |
| `rest_diff` | `home_rest − away_rest` |
| `home_form_rundiff` | Home team's rolling mean run differential over recent games |
| `away_form_rundiff` | Away team's rolling mean run differential |
| `home_form_winpct` | Home team's rolling win percentage |
| `away_form_winpct` | Away team's rolling win percentage |
| `month` | Calendar month of the game (1–12) |
| `playoff` | 1 if a postseason game, else 0 |

**Label:** `home_win` — 1 if the home team won, 0 otherwise.

**Not fed to the model:** metadata columns (`game_pk`, `date`, `season`, `home_team`,
`away_team`) and the regression target `run_margin` (used only by the margin model).

The recent-form window size defaults to 10 games (`features.recent_form_window`).

## Training process

### Model families

Two **classifier** families are compared:

- **Logistic regression** — L2-regularized, with feature standardization.
- **LightGBM** — gradient-boosted trees with conservative regularization defaults.

When enabled (`ml.margin.enabled`), a parallel **run-margin** model is also trained
(Ridge regression vs. LightGBM regressor). It predicts `home_score − away_score` and
derives win probability from a fitted Normal residual distribution.

### Walk-forward cross-validation

Models are evaluated with **expanding-window cross-validation by season**:

- Train on all seasons **before** year *Y*.
- Test on season *Y*.
- Repeat for each test season (requires at least 2 prior training seasons).

This mimics real deployment: the model never sees future seasons during evaluation.

### Per-fold training loop

For each model family and each CV fold:

1. **Fit a calibrated model** on the training seasons (`fit_calibrated_model`):
   - Sort games by date.
   - Fit the base estimator (logistic or LightGBM) on the earlier 80% of training rows.
   - Hold out the last 20% (minimum 50 games) and fit a **probability calibrator**
     (Platt sigmoid by default; isotonic is also supported) on the base model's raw
     probabilities.
   - If the training set has fewer than 200 games, calibration is skipped.

2. **Predict** calibrated home-win probabilities on the held-out test season.

3. **Score** pooled out-of-sample predictions with:
   - **Log loss** — penalizes confident wrong predictions; lower is better. A coin flip
     scores ~0.693.
   - **Brier score** — mean squared error between predicted probability and the binary
     outcome; lower is better.

   > The [Brier score](https://en.wikipedia.org/wiki/Brier_score) measures how far
   > predicted probabilities are from the actual 0/1 result.

### Model selection and final fit

After CV completes for all families:

1. The family with the **lowest pooled log loss** is selected.
2. A final calibrated model is **refit on all available training data** (same
   calibration split).
3. The model, feature column list, Elo parameters, and CV report are saved to the model
   registry (`data/models/`).

The margin model follows the same expanding-window CV but is selected by derived win-prob
log loss, then refit on all data.

### Example CV output

```
[cv]  logistic: log_loss=0.6768 brier=0.2419 (n=14800)
[cv]  lightgbm: log_loss=0.6841 brier=0.2455 (n=14800)
[cv] selected: logistic (log_loss=0.6768)
```

### Configuration

Key training knobs in `config.yaml`:

| Setting | Purpose |
|---|---|
| `ml.cv.min_train_seasons` | Minimum prior seasons before a fold is scored |
| `ml.logistic.C` | Inverse L2 strength for logistic regression |
| `ml.lightgbm.*` | Tree count, learning rate, regularization |
| `ml.calibration` | `sigmoid`, `isotonic`, or `none` |
| `features.include.*` | Toggle individual feature groups on/off |

To trial variants without editing `config.yaml`, add presets to `experiments.yaml` and
run `mlbfc compare`.
