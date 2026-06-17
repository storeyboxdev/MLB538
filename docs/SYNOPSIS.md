# MLB Forecaster — How It Works

## The goal

Forecast Major League Baseball the way FiveThirtyEight did: rate every team, predict
every game, and roll those predictions up into season-long **playoff, division, pennant,
and World Series odds** — updated as games are played.

The twist versus a classic Elo model: Elo here is not the final predictor. It is a
**feature generator** feeding a trained machine-learning model. The system learns from
history, validates itself honestly, and self-corrects as new results arrive.

## The data

One source: the **official, free MLB Stats API** (`statsapi.mlb.com`). No scraping of
fragile web pages, no API key. We pull schedules, final scores, probable/actual starting
pitchers, and pitcher box-score lines. Responses are cached on disk — completed games
never change — so re-runs are fast and gentle on the API.

## The model — three layers

**1. Elo engine (the "physics").**
Every team carries an Elo rating that updates after each game. Win probability follows
the standard logistic curve. On top of the textbook formula we add the same refinements
538 used:
- **Home-field advantage** (~24 Elo points).
- A **margin-of-victory multiplier** so blowouts move ratings more than one-run games —
  with an *autocorrelation damper* so heavy favorites can't farm rating points by
  beating up weak teams.
- A **starting-pitcher adjustment**: each starter gets a rolling, recency-weighted
  "game score" (from strikeouts, outs, walks, hits, runs, home runs). A good starter
  temporarily raises his team's effective rating for that game; a poor one lowers it.
- **Preseason reversion**: ratings partially revert toward the mean each offseason.

Crucially, the Elo *hyperparameters* (the knobs above) are **fit to historical data**,
not hand-set — we let the data choose them.

**2. ML predictor (the "judgment").**
For every game we snapshot the **pre-game state** — Elo gap, pitcher adjustments, days
of rest, recent form, month — and train a model to map that to a win probability. We
run two model families (regularized **logistic regression** and **LightGBM** gradient
boosting) and keep whichever predicts better. The output is **calibrated**, meaning when
the model says 60% it really happens about 60% of the time.

**3. Season simulation (the "forecast").**
Using current ratings, we **Monte Carlo the rest of the season** thousands of times —
playing out every remaining game, then the full 12-team playoff bracket — and count how
often each team makes the playoffs, wins its division, takes the pennant, and wins it
all.

## How it self-corrects

Two feedback loops:
- **Online Elo**: every new final game nudges ratings immediately — the model is never
  more than a day stale.
- **Periodic retraining**: on a schedule, the ML layer is refit on a rolling window of
  recent seasons and the Elo knobs are re-optimized, with rolling accuracy logged so
  drift is visible. One command (`update`) runs the whole loop and can be scheduled.

## How we know it's honest (not overfit)

This is the part that matters for trust:
- **Walk-forward validation**: we only ever test on seasons the model has *not* seen
  (train on years ≤ Y, test on Y+1). No peeking at the future.
- **Leakage guard**: every feature is strictly pre-game; an automated test fails the
  build if outcome data sneaks into the inputs.
- **Backtest comparison**: we score plain Elo vs. the ML models head-to-head on the same
  out-of-sample games, with a calibration table.

## What we've found so far

- **Calibration is excellent** — predicted probabilities match observed results across
  the board.
- **It reproduces reality**: in 2024 it gave the Dodgers the top World Series odds (they
  won), and its simulated playoff field matched the actual one.
- **A reality check on baseball**: MLB is the hardest of the major sports to predict —
  even the best models barely beat a coin flip on a single game. Elo-plus-pitcher and the
  trained model currently sit at near-identical accuracy, because team-level signal alone
  is close to the ceiling.
- **Pitcher data adds real lift**: where we have full pitcher box scores, the starting-
  pitcher adjustment measurably beats team-only ratings. Loading that data across all
  training seasons is our highest-value next step.

## Roadmap

1. **Full pitcher box-score history** across all seasons (in progress) — expected to be
   where the ML model pulls clearly ahead of plain Elo.
2. **Richer features**: ballpark factors, travel/fatigue, bullpen strength, weather.
3. **Run-margin modeling** for sharper game and series probabilities.
4. **Presentation layer**: a simple dashboard for ratings history and live odds.

## In one breath

> We pull official MLB data, rate teams with a pitcher-aware Elo system, feed those
> ratings into a calibrated machine-learning model, and simulate the rest of the season
> thousands of times to produce playoff and championship odds — validated only on seasons
> the model never trained on, and refreshed automatically as games are played.
