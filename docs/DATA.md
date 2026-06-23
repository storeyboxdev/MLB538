# Data: the MLB Stats API

This project sources all of its data from the **MLB Stats API**
(`https://statsapi.mlb.com/api`). The API is free, needs no key, and serves JSON.
Completed-game data is immutable, so raw responses are cached on disk
(`data/raw/api/<sha1>.json`) keyed by a hash of the full request URL — see
[`mlb_forecaster/api/client.py`](mlb_forecaster/api/client.py).

This document describes **what data we fetch and what fields we keep**. The API
exposes far more than we use; the goal here is to map what is actually available
to the columns that end up in `data/processed/games_<season>.csv`.

---

## Endpoints we use

| Endpoint | Purpose | Code |
| --- | --- | --- |
| `GET v1/teams?sportId=1&season=<yyyy>` | Team metadata for the season | [`api/teams.py`](mlb_forecaster/api/teams.py) |
| `GET v1/schedule?sportId=1&startDate=…&endDate=…&hydrate=probablePitcher,linescore,team` | Full season schedule + results | [`api/schedule.py`](mlb_forecaster/api/schedule.py) |
| `GET v1/game/{game_pk}/boxscore` | Per-game starter pitching lines | [`api/games.py`](mlb_forecaster/api/games.py) |

`sportId=1` restricts results to MLB (the API also serves minors, winter ball,
spring training, etc.).

---

## 1. Teams — `v1/teams`

Returns one record per club. We keep:

| Field | Source JSON | Notes |
| --- | --- | --- |
| `abbr` | `abbreviation` | e.g. `NYY` — the primary key used everywhere downstream |
| `name` | `name` | full club name |
| `league` | `league.id` → `AL`/`NL` | `103` = AL, `104` = NL |
| `league_id` | `league.id` | raw id |
| `division` | `division.name` | e.g. "American League East" |
| `division_id` | `division.id` | raw id |

The team id → abbreviation map is used to label every game and to drop
non-MLB opponents (e.g. exhibition games vs. minor-league clubs).

---

## 2. Schedule — `v1/schedule`

One request pulls an entire season (`startDate=<yyyy>-03-01` to
`<yyyy>-11-30`). Hydrated with `probablePitcher`, `linescore`, and `team`.

We keep only these `gameType`s:

| Code | Meaning | `playoff` flag |
| --- | --- | --- |
| `R` | Regular season | `False` |
| `F` | Wild Card | `True` |
| `D` | Division Series | `True` |
| `L` | League Championship Series | `True` |
| `W` | World Series | `True` |

Spring training (`S`) and exhibition (`E`) games are skipped.

Fields parsed per game:

| Column | Source JSON | Notes |
| --- | --- | --- |
| `game_pk` | `gamePk` | unique game id |
| `date` | `officialDate` (fallback `gameDate`) | calendar date |
| `season` | `season` | |
| `game_type` | `gameType` | see table above |
| `home_team` / `away_team` | `teams.{home,away}.team.id` → abbr | |
| `home_score` / `away_score` | `teams.{home,away}.score` | `null` until played |
| `status` | `status.abstractGameState` | `Final`, `Scheduled`, `In Progress`, … |
| `home_pitcher_id` / `away_pitcher_id` | `teams.{side}.probablePitcher.id` | probable starter |
| `home_pitcher` / `away_pitcher` | `teams.{side}.probablePitcher.fullName` | |
| `venue_id` | `venue.id` | |
| `double_header` | `doubleHeader` | `N`, `Y`, `S` |

---

## 3. Boxscore — `v1/game/{game_pk}/boxscore`

Fetched only for **Final** games (skippable with `fetch_boxscores=False`). We
extract the **starting pitcher's line** for each side — the starter is the first
entry in `teams.{side}.pitchers`, read from
`teams.{side}.players.ID<pid>.stats.pitching`.

| Column | Source JSON (`…pitching`) | Meaning |
| --- | --- | --- |
| `home_outs` / `away_outs` | `outs` | outs recorded (IP × 3) |
| `home_hits` / `away_hits` | `hits` | hits **allowed** by the starter |
| `home_runs` / `away_runs` | `runs` | runs **allowed** |
| `home_earned_runs` / `away_earned_runs` | `earnedRuns` | earned runs allowed |
| `home_walks` / `away_walks` | `baseOnBalls` | walks allowed |
| `home_strikeouts` / `away_strikeouts` | `strikeOuts` | strikeouts recorded |
| `home_home_runs` / `away_home_runs` | `homeRuns` | home runs allowed |

> ⚠️ **Important semantics:** these `*_hits`, `*_runs`, etc. columns are the
> **starting pitcher's box-score line**, not team batting totals. `home_hits`
> means "hits surrendered by the home team's starter," *not* "hits the home team
> got." They exist to compute a Bill James starter game score for the Elo
> pitcher adjustment. We do **not** currently ingest team batting stats.

Also derived during ingestion:

| Column | Meaning |
| --- | --- |
| `home_starter_id` / `away_starter_id` | actual starter id from the boxscore, falling back to the probable-pitcher id when no boxscore was fetched |

---

## Processed schema

After ingestion ([`api/games.py`](mlb_forecaster/api/games.py)), each season is
written to `data/processed/games_<season>.csv`. One row per `game_pk`
(de-duplicated — the schedule occasionally returns a played row plus a
postponed/placeholder row; the played one wins). Columns are the union of the
schedule fields, the flattened starter lines, plus `playoff`,
`home_starter_id`, and `away_starter_id`.

The canonical dataclasses are documented in
[`mlb_forecaster/data/models.py`](mlb_forecaster/data/models.py) (`Game`,
`PitcherLine`).

---

## Data we do *not* currently pull (but the API offers)

The Stats API exposes much more that this project could ingest later:

- **Team & player batting/fielding stats** (`stats?stats=season&group=hitting`, per-game
  `boxscore.teams.{side}.teamStats`) — actual team hits, runs, OBP, etc.
- **Play-by-play** (`v1/game/{pk}/playByPlay`) — pitch-level / event-level data.
- **Live feed** (`v1.1/game/{pk}/feed/live`) — full real-time game state.
- **Standings** (`v1/standings`).
- **Rosters / player bios** (`v1/teams/{id}/roster`, `v1/people/{id}`).
- **Venues, weather, attendance** (additional schedule hydrations).

If a future feature needs real team batting totals (e.g. "hits per game by
team"), the boxscore `teams.{side}.teamStats.batting.hits` field is the place to
get it — the current `*_hits` columns will not answer that question.
