"""Optional Google Cloud Storage sink for processed data.

Uploads each season's game log as Parquet so BigQuery can ingest it (typed,
compressed, network-cheap). Enabled by setting ``data.gcs_bucket`` in config or
the ``MLB538_GCS_BUCKET`` environment variable. Requires the ``gcp`` extra
(``pip install -e ".[gcp]"``).

On a GCP VM, authentication is automatic via the instance service account
(Application Default Credentials) — no key file needed. The VM must have a
storage write scope; see docs/DEPLOY_GCP.md.
"""

from __future__ import annotations

import io
import os

import pandas as pd

from ..config import Config


def bucket_name(config: Config) -> str | None:
    """Resolve the target bucket (env var overrides config)."""
    return os.environ.get("MLB538_GCS_BUCKET") or config.raw["data"].get("gcs_bucket")


def prefix(config: Config) -> str:
    p = os.environ.get("MLB538_GCS_PREFIX") or config.raw["data"].get("gcs_prefix", "mlb538")
    return p.strip("/")


def is_enabled(config: Config) -> bool:
    return bool(bucket_name(config))


def _client():
    # Deferred import so the package works without the gcp extra installed.
    from google.cloud import storage
    return storage.Client()


def upload_bytes(data: bytes, bucket: str, blob_name: str,
                 content_type: str | None = None) -> str:
    blob = _client().bucket(bucket).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket}/{blob_name}"


def upload_file(local_path, bucket: str, blob_name: str) -> str:
    _client().bucket(bucket).blob(blob_name).upload_from_filename(str(local_path))
    return f"gs://{bucket}/{blob_name}"


def upload_games(df: pd.DataFrame, config: Config, season: int) -> str:
    """Upload a season's games as Parquet to gs://<bucket>/<prefix>/processed/."""
    bucket = bucket_name(config)
    blob_name = f"{prefix(config)}/processed/games_{season}.parquet"
    out = df.copy()
    if "date" in out.columns:
        # Coerce ISO strings to datetime so BigQuery infers a TIMESTAMP column.
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    return upload_bytes(buf.getvalue(), bucket, blob_name, "application/octet-stream")
