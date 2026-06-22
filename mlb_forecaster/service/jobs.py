"""A minimal single-worker background job manager.

Heavy pipeline operations (scrape, train, backtest, ...) can take minutes, which
is too long for a synchronous tool/HTTP call. They are submitted here instead:
each returns a job id immediately, and the caller polls for status/result. A
single worker thread guarantees jobs run one at a time, so two heavy jobs never
race on the same on-disk artifacts (model registry, processed CSVs, output JSON).
"""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

# A job body receives a ``log`` callback (mirrors the pipeline functions' ``log``
# parameter) and returns a JSON-serializable result.
JobBody = Callable[[Callable[[str], None]], Any]

_MAX_LOG_LINES = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """One unit of background work and its lifecycle state."""

    id: str
    name: str
    params: dict
    status: str = "queued"  # queued | running | succeeded | failed
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        """Append a log line (bounded), used as the pipeline ``log`` callback."""
        line = str(msg)
        self.logs.append(line)
        if len(self.logs) > _MAX_LOG_LINES:
            del self.logs[0]

    def to_dict(self, include_result: bool = True) -> dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "logs": list(self.logs),
        }
        if include_result:
            d["result"] = self.result
        return d


class JobManager:
    """Submit and track background jobs, executed one at a time."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # max_workers=1 => strict serialization of heavy jobs.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlbfc-job")

    def submit(self, name: str, body: JobBody, params: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, params=params or {})
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, body)
        return job

    def _run(self, job: Job, body: JobBody) -> None:
        job.status = "running"
        job.started_at = _now()
        try:
            job.result = body(job.log)
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            job.error = f"{type(exc).__name__}: {exc}"
            job.log("[error] " + traceback.format_exc())
            job.status = "failed"
        finally:
            job.finished_at = _now()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
