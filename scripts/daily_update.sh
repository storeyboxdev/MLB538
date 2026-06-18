#!/usr/bin/env bash
#
# Daily incremental update: pull new games, refresh ratings + forecast, reload BigQuery.
#
# Designed to run from cron on a host with PERSISTENT disk so the API cache
# (data/raw/) survives between runs — that's what makes the scrape incremental
# (only newly-finished games are downloaded; completed games are cache hits).
#
# Required env:
#   MLB538_GCS_BUCKET   GCS bucket name for parquet uploads (also enables the GCS sink)
# Optional env:
#   MLB538_BQ_DATASET   BigQuery dataset name      (default: mlb538)
#   MLB538_GCS_PREFIX   object prefix in the bucket (default: mlb538)
#   MLB538_SEASON       season to update           (default: current calendar year)
#
# Example cron entry (7:00 AM daily):
#   0 7 * * * MLB538_GCS_BUCKET=my-bucket /home/USER/MLB538/scripts/daily_update.sh \
#             >> /home/USER/MLB538/daily.log 2>&1
#
set -euo pipefail

# Resolve repo root (parent of this script's directory) and cd there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Activate the virtualenv if one exists.
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

: "${MLB538_GCS_BUCKET:?Set MLB538_GCS_BUCKET to your bucket name}"
DATASET="${MLB538_BQ_DATASET:-mlb538}"
PREFIX="${MLB538_GCS_PREFIX:-mlb538}"
SEASON="${MLB538_SEASON:-$(date +%Y)}"
B="gs://${MLB538_GCS_BUCKET}/${PREFIX}"

echo "=== $(date -u +%FT%TZ) daily update | season=$SEASON bucket=$MLB538_GCS_BUCKET dataset=$DATASET ==="

# 1. Incremental scrape of the current season (cache => only new games) + retrain on
#    cadence + online Elo. Uploads the season's games parquet to the bucket.
mlbfc update --season "$SEASON"

# 2. Rebuild ratings (all seasons) and the current-season forecast; both upload parquet.
mlbfc rate
mlbfc forecast --season "$SEASON"

# 3. Reload BigQuery from the bucket. --replace rebuilds each table from all parquet
#    files (tiny dataset, so a full reload is effectively free).
bq load --source_format=PARQUET --replace --autodetect "${DATASET}.games"    "${B}/processed/games_*.parquet"
bq load --source_format=PARQUET --replace --autodetect "${DATASET}.ratings"  "${B}/ratings/ratings.parquet"
bq load --source_format=PARQUET --replace --autodetect "${DATASET}.forecast" "${B}/forecast/forecast_*.parquet"

echo "=== $(date -u +%FT%TZ) daily update complete ==="
