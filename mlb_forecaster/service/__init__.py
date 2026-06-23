"""Service layer: a shared core plus FastAPI and MCP front-ends.

Both front-ends (``api.py`` for HTTP/REST, ``mcp_server.py`` for MCP) are thin
adapters over :class:`mlb_forecaster.service.core.Service`, which wraps the
existing ``pipeline`` functions. Fast operations (model info, presets, forecast)
run synchronously; heavy ones (scrape, train, fit-elo, backtest, compare,
update) are dispatched to a single-worker background :class:`JobManager` so they
serialize and never clobber each other's on-disk artifacts.
"""

from .core import Service
from .jobs import Job, JobManager

__all__ = ["Service", "Job", "JobManager"]
