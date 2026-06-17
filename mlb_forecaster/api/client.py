"""HTTP client for the MLB Stats API with on-disk caching and retry.

The MLB Stats API (``statsapi.mlb.com``) is free and needs no key. Completed-game
data is immutable, so we cache raw JSON responses on disk keyed by a hash of the
full URL. This keeps re-runs fast and gentle on the API.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests

from ..config import Config


class MLBStatsClient:
    def __init__(self, config: Config):
        self.cfg = config.raw["api"]
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.cache_dir = config.raw_dir
        self.cache_enabled = self.cfg.get("cache_enabled", True)
        self.timeout = self.cfg.get("request_timeout", 30)
        self.max_retries = self.cfg.get("max_retries", 4)
        self.backoff = self.cfg.get("backoff_factor", 0.5)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "mlb-forecaster/0.1 (+local)"})

    # --- caching --------------------------------------------------------
    def _cache_key(self, url: str) -> Path:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / "api" / f"{digest}.json"

    def _read_cache(self, url: str) -> Optional[Any]:
        if not self.cache_enabled:
            return None
        path = self._cache_key(url)
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _write_cache(self, url: str, payload: Any) -> None:
        if not self.cache_enabled:
            return
        path = self._cache_key(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    # --- requests -------------------------------------------------------
    def get(self, endpoint: str, params: Optional[dict[str, Any]] = None,
            use_cache: bool = True) -> Any:
        """GET an endpoint (e.g. ``v1/schedule``) and return parsed JSON.

        ``use_cache=False`` forces a fresh fetch (used for live/in-progress data).
        """
        params = params or {}
        query = urlencode(params, doseq=True)
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query:
            url = f"{url}?{query}"

        if use_cache:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        payload = self._request_with_retry(url)
        if use_cache:
            self._write_cache(url, payload)
        return payload

    def _request_with_retry(self, url: str) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} for {url}")
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                sleep_s = self.backoff * (2 ** attempt)
                time.sleep(sleep_s)
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts") from last_exc
