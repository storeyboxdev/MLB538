from __future__ import annotations

import os
from typing import Optional

import requests
import google.auth
from google.auth.transport.requests import Request
from google.cloud import bigquery
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

# API configuration
API_BASE_URL = os.getenv("MLB_API_BASE_URL", os.getenv("MLB_API_URL", "http://34.173.102.166:8000"))
API_TIMEOUT = float(os.getenv("MLB_API_TIMEOUT", "180"))

API_URL = API_BASE_URL.rstrip("/")
_TIMEOUT = API_TIMEOUT
_TOKEN = os.getenv("MLB_API_TOKEN")
_session = requests.Session()
if _TOKEN:
    _session.headers["Authorization"] = f"Bearer {_TOKEN}"

# BigQuery configuration
PROJECT_ID = "qwiklabs-asl-02-03bf2b8329ea"
BIGQUERY_MCP_URL = "https://bigquery.googleapis.com/mcp"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID


def get_bigquery_mcp_toolset() -> MCPToolset:
    """
    Create an MCPToolset connected to Google's managed BigQuery MCP server.
    """
    # Get OAuth credentials
    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    credentials.refresh(Request())
    oauth_token = credentials.token

    # Use environment project if available
    if PROJECT_ID:
        project_id = PROJECT_ID

    # Create headers with OAuth token
    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "x-goog-user-project": project_id,
    }

    # Create the MCPToolset
    tools = MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=BIGQUERY_MCP_URL,
            headers=headers,
        )
    )

    print(f"[BigQueryTools] MCP Toolset configured for project: {project_id}")

    return tools


def _get_headers() -> dict:
    """Build request headers."""
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _get(path: str, params: Optional[dict] = None) -> dict:
    r = _session.get(
        f"{API_URL}{path}",
        params=params,
        headers=_get_headers(),
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    # requests' json= sets Content-Type: application/json (avoids the 422 a raw
    # form-encoded body would cause).
    r = _session.post(f"{API_URL}{path}", json=body, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def forecast(season: int, sims: Optional[int] = None) -> dict:
    """Run a season forecast: simulate the rest of the season and return each
    team's projected wins and playoff/division/pennant/World Series odds.
    """
    return _post("/forecast", {"season": season, "sims": sims})

def get_health() -> dict:
    """Check that the forecaster API is up and reachable.
    Returns:
        dict: Includes 'status' and 'available' boolean.
    """
    print(f"--- Tool: get_health called for {API_URL} ---")
    try:
        data = _get("/health")
        return {
            "available": True,
            "status": data.get("status", "unknown"),
            "url": API_URL,
            "response": data,
        }
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        return {
            "available": False,
            "error": "http_error",
            "status_code": status_code,
            "message": f"API returned status {status_code}" if status_code else "HTTP error",
            "url": API_URL,
        }
    except requests.ConnectionError:
        return {
            "available": False,
            "error": "connection_error",
            "message": f"Could not connect to API at {API_URL}",
            "url": API_URL,
        }
    except requests.Timeout:
        return {
            "available": False,
            "error": "timeout",
            "message": f"API request timed out after {_TIMEOUT}s",
            "url": API_URL,
        }
    except Exception as e:
        return {
            "available": False,
            "error": "unknown",
            "message": str(e),
            "url": API_URL,
        }
