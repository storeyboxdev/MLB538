---
marp: true
title: MLB Forecaster — A 538-style, ML-driven model
paginate: true
theme: default
---

<!--
Speaker notes appear in these HTML comment blocks. To render as slides:
  npm i -g @marp-team/marp-cli
  marp docs/PRESENTATION.md -o presentation.pdf   (or --html, or --pptx)
Each "---" starts a new slide.
-->

# Forecasting Major League Baseball

### A FiveThirtyEight-style, machine-learning model

Rate every team · predict every game · simulate the season · publish the odds

<!--
One-liner to open with: "We built the kind of model 538 used to run for baseball —
team ratings that update game by game, rolled up into playoff and World Series odds —
but with a machine-learning layer on top and an honest way to prove it works."
Keep this slide up for ~30 seconds while people settle.
-->

---

## The question we're answering

> Given everything we know today, **what are each team's chances** of making the
> playoffs, winning the division, and winning the World Series?

- Updated continuously as games are played
- Grounded in data, not gut
- Honest about its own uncertainty

<!--
Frame it as a decision/communication tool, not a betting tool. The output is a
probability table the whole org can rally around. Emphasize "honest about
uncertainty" — that's our differentiator and we'll back it up later.
-->

---

## The big idea

**Elo is the engine. Machine learning is the driver.**

- A pitcher-aware **Elo rating** measures team strength and updates after every game
- Those ratings become **features** for a trained, calibrated ML model
- The model's win probabilities feed a **Monte Carlo simulation** of the season

We don't just hand-tune a formula — the system **learns its settings from history**
and **corrects itself** as new results arrive.

<!--
The mental model: Elo is the "physics" (a principled rating that moves with results),
ML is the "judgment" (learns how to turn ratings + context into a probability),
simulation is the "forecast." Stress that nothing is hand-set — parameters are fit.
-->

---

## Architecture at a glance

```
 MLB Stats API        Elo engine            ML predictor         Simulation
 (official, free) --> (team ratings,   -->  (logistic / GBM, --> (10k seasons -->
  schedules,           pitcher-aware,        calibrated           + playoff
  scores, pitchers)    fit to history)       probabilities)       bracket)
                                                                      |
                                                                      v
                                                         Playoff / WS odds (JSON)
```

One command line drives each stage; one `update` command runs the whole loop daily.

<!--
Walk left to right. Note the single clean data source (no scraping fragile web pages).
Each box is a real module with tests. The arrows are the only contract between stages,
which makes it easy to swap or improve any one box.
-->

---

## Layer 1 — The Elo engine ("the physics")

Each team carries a rating; it updates after every game. Refinements (same ones 538 used):

- **Home-field advantage** (~22 Elo points, *fit from data*)
- **Margin-of-victory** multiplier — blowouts move ratings more...
- ...with an **autocorrelation damper** so favorites can't farm points off weak teams
- **Starting-pitcher adjustment** — a rolling "game score" per starter
- **Off-season reversion** toward the mean

> The knobs above are **optimized against history**, not guessed.

<!--
Don't go deep on math. The point: it's the textbook Elo curve plus four baseball-aware
corrections, and critically, we *fit* the knobs rather than hand-setting them. Tease
the pitcher adjustment here — it becomes a key result later.
-->

---

## Layer 2 — The ML predictor ("the judgment")

For every game we snapshot the **pre-game state**:

`Elo gap · pitcher edge · days of rest · recent form · month`

- Train **two model families** — logistic regression and gradient boosting (LightGBM)
- Keep whichever predicts better out-of-sample
- Output is **calibrated**: when it says 60%, it happens ~60% of the time

<!--
Why two models? Logistic = stable, interpretable baseline; LightGBM = catches
nonlinearities. We let the data pick. Calibration is the feature people trust — a
60% that means 60%. We verify calibration on held-out data (next section).
-->

---

## Layer 3 — Season simulation ("the forecast")

From today's ratings, **play out the rest of the season 10,000 times**:

- Simulate every remaining game
- Run the full **12-team playoff bracket** (3 division winners + 3 wild cards/league)
- Tally how often each team makes the playoffs / wins the division / pennant / title

The result is a clean odds table — the thing you actually present.

<!--
This is what turns "team X is rated 1560" into "team X has a 25% chance to win it all."
Monte Carlo handles all the messy schedule and bracket interactions automatically.
Mention it's fast — thousands of full seasons in seconds.
-->

---

## How it self-corrects

Two feedback loops keep it fresh and honest:

1. **Online Elo** — every final nudges ratings immediately; never more than a day stale
2. **Periodic retraining** — the ML layer refits on a rolling window; Elo knobs
   re-optimize; rolling accuracy is logged so **drift is visible**

A single scheduled `update` command runs the entire loop.

<!--
"Self-correcting" has a concrete meaning here: ratings update online, and the learned
model + parameters get refreshed on a cadence. We log rolling error so if the model
starts drifting, we see it in the metrics, not in a surprise.
-->

---

## How we know it's honest (not overfit)

This is the part that earns trust:

- **Walk-forward validation** — only ever tested on seasons it *never trained on*
  (train ≤ year Y, test Y+1)
- **Leakage guard** — an automated test fails the build if any outcome data leaks
  into the inputs
- **Head-to-head backtest** — plain Elo vs. ML on the *same* out-of-sample games,
  with a calibration table

<!--
Anticipate the skeptic: "Of course it fits the past." Our answer: we never test on
data the model saw, a unit test enforces no leakage, and we benchmark against a simple
Elo baseline so we can prove the added complexity earns its keep. This slide is the
credibility anchor.
-->

---

## Results — accuracy (8 seasons, ~14,800 out-of-sample games)

| Model | Log loss | Brier |
|---|---|---|
| Elo (team only) | 0.6765 | 0.2418 |
| **Elo + pitcher adjustment** | **0.6754** | **0.2413** |
| Logistic regression (ML) | 0.6768 | 0.2419 |
| LightGBM (ML) | 0.6840 | 0.2455 |

*(Lower is better. A coin flip = 0.6931 log loss.)*

**Calibration is excellent** — predicted probabilities match observed results across
every bucket.

<!--
Read the table honestly. Pitcher-adjusted Elo is best; the ML logistic is at parity,
fractionally behind; LightGBM overfits and was correctly rejected by model selection.
The headline is NOT "ML crushes Elo." The headline is "well-calibrated, and pitching
demonstrably matters." Set up the next slide.
-->

---

## Results — it reproduces reality

- **2024:** gave the **Dodgers the top World Series odds** — they won it
- Its simulated **playoff field matched the actual one**
- Tight races show up as tight odds (e.g., a three-way ~66% wild-card scramble)

<!--
Sanity beats statistics for a lay audience. The model independently landing on the
eventual champion and the correct playoff field is the most persuasive proof for
non-statisticians. Use this as the "it just works" moment.
-->

---

## The pitcher result (why this matters)

Given real starting-pitcher box scores, the model **taught itself** that pitching matters:

- The fitter **raised the pitcher weight 1.0 → 1.6** on its own
- **Pitcher-adjusted Elo beats team-only Elo** across all 14,800 test games

> Starting pitching was the single biggest lever — and it's now in the model.

<!--
This is the satisfying "the model discovered something true" slide. We didn't tell it
pitching matters; the optimization, handed the data, increased the pitcher weight and
the accuracy improved. Concrete evidence the framework responds to real signal.
-->

---

## The honest takeaway

**Baseball is the hardest major sport to predict.** Even the best public models barely
beat a coin flip on a single game.

- A finely-tuned, pitcher-aware Elo is **near the single-game ceiling**
- Our ML model **matches** it and is **well-calibrated**
- The ML framework's real payoff: it's a calibrated floor that will **automatically
  capitalize on richer features as we add them**

<!--
Don't oversell. The defensible claim: we're at the practical ceiling for team-level
signal, we're honest and calibrated, and we've built the on-ramp for improvement.
This honesty is itself a selling point with a technical audience.
-->

---

## Roadmap — where the edge comes from next

Signal that Elo *doesn't* already capture:

1. **Bullpen / reliever quality**, lineup construction, day-of injuries
2. **Statcast team quality** (xwOBA, expected run prevention) — separates lucky from good
3. **Ballpark factors, weather, travel / fatigue**
4. **Run-margin modeling** for sharper game & series probabilities
5. **Dashboard** for live odds and rating history

<!--
Frame the roadmap as "orthogonal signal" — features that add information beyond what
team W-L Elo already encodes. That's where ML can pull ahead of Elo. Prioritize
bullpen + Statcast; those are the highest-value, most-orthogonal additions.
-->

---

## In one breath

> We pull official MLB data, rate teams with a pitcher-aware Elo system, feed those
> ratings into a calibrated machine-learning model, and simulate the rest of the season
> thousands of times to produce championship odds — validated only on seasons the model
> never trained on, and refreshed automatically as games are played.

**Built · tested · validated · honest.**

<!--
Closing line. Land the four words: built, tested, validated, honest. Then open for
questions. Likely Qs: "Can it beat Vegas?" (no claim — different problem, and lines
embed juice/market info); "Why not deep learning?" (data volume is small, ~2,400
games/yr; trees/logistic are the right tools and stay calibrated); "How often does it
update?" (daily, one command).
-->

---

## Appendix — anticipated questions

- **Can it beat the betting market?** Different problem; betting lines embed market
  information and vig. We're a forecasting/communication tool, not a wagering system.
- **Why not a neural network?** ~2,400 games/season is small data; gradient boosting and
  logistic regression are the right-sized tools and stay well-calibrated.
- **How fresh is it?** Online Elo updates every game; full retrain runs on a schedule
  via one `update` command.
- **What's the data source?** The official, free MLB Stats API — no scraping, no key.

<!--
Keep this slide hidden unless asked. These are the four questions that always come up.
Have crisp answers ready and you'll close strong.
-->
