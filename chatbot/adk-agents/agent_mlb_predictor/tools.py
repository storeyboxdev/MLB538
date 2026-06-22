import os
import requests
import google.auth
from google.auth.transport.requests import Request
from google.cloud import bigquery
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

# API configuration
API_BASE_URL = os.getenv("MLB_API_BASE_URL", "http://35.238.218.28:8000")
API_TIMEOUT = 20

# BigQuery Configuration
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


def get_health() -> dict:
    """Check the health status of the MLB API endpoint.
    Returns:
        dict: Includes 'status' and 'available' boolean.
    """
    print(f"--- Tool: get_health called for {API_BASE_URL} ---")
    
    url = f"{API_BASE_URL}/health"
    
    try:
        resp = requests.get(
            url,
            headers=_get_headers(),
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "available": True,
            "status": data.get("status", "unknown"),
            "url": API_BASE_URL,
            "response": data,
        }
    except requests.HTTPError as e:
        return {
            "available": False,
            "error": "http_error",
            "status_code": e.response.status_code,
            "message": f"API returned status {e.response.status_code}",
            "url": API_BASE_URL,
        }
    except requests.ConnectionError:
        return {
            "available": False,
            "error": "connection_error",
            "message": f"Could not connect to API at {API_BASE_URL}",
            "url": API_BASE_URL,
        }
    except requests.Timeout:
        return {
            "available": False,
            "error": "timeout",
            "message": f"API request timed out after {API_TIMEOUT}s",
            "url": API_BASE_URL,
        }
    except Exception as e:
        return {
            "available": False,
            "error": "unknown",
            "message": str(e),
            "url": API_BASE_URL,
        }

# TODO: connect to database (ex: BigQuery MCP)
# Mock prediction data by year -> team
MOCK_PREDICTION_DB: dict[str, dict[str, dict[str, float]]] = {
    "2024": {
        "Dodgers": {
            "make_playoffs_chance": 0.93,
            "win_division_chance": 0.72,
            "win_world_series_chance": 0.18,
            "prediction_for_team_given_a_year": 0.81,
        },
        "Giants": {
            "make_playoffs_chance": 0.47,
            "win_division_chance": 0.21,
            "win_world_series_chance": 0.04,
            "prediction_for_team_given_a_year": 0.44,
        },
    },
    "2025": {
        "Dodgers": {
            "make_playoffs_chance": 0.91,
            "win_division_chance": 0.69,
            "win_world_series_chance": 0.17,
            "prediction_for_team_given_a_year": 0.79,
        },
        "Giants": {
            "make_playoffs_chance": 0.51,
            "win_division_chance": 0.24,
            "win_world_series_chance": 0.05,
            "prediction_for_team_given_a_year": 0.48,
        },
    },
}

# Mock ELO data by year -> team
MOCK_ELO_DB: dict[str, dict[str, float]] = {
    "2024": {
        "Dodgers": 1650,
        "Giants": 1520,
        "Yankees": 1680,
        "Astros": 1600,
    },
    "2025": {
        "Dodgers": 1680,
        "Giants": 1540,
        "Yankees": 1700,
        "Astros": 1620,
    },
}

# Mock World Series data
MOCK_WORLD_SERIES_DB: dict[str, dict[str, str]] = {
    "2024": {"winner": "Dodgers", "loser": "Yankees"},
    "2025": {"winner": "Dodgers", "loser": "Astros"},
}

# Mock trade deadline data
MOCK_TRADE_DEADLINE_DB: dict[str, dict[str, dict[str, dict[str, float]]]] = {
    "2024": {
        "Dodgers": {"pre": 1620, "post": 1650},
        "Giants": {"pre": 1500, "post": 1520},
    },
    "2025": {
        "Dodgers": {"pre": 1650, "post": 1680},
        "Giants": {"pre": 1520, "post": 1540},
    },
}

# Mock hot/cold data by year-month
MOCK_HOT_COLD_DB: dict[str, list[dict]] = {
    "2024-06": [
        {"team": "Dodgers", "elo_change_last_30_days": 45, "trend": "hot"},
        {"team": "Giants", "elo_change_last_30_days": -30, "trend": "cold"},
        {"team": "Yankees", "elo_change_last_30_days": 20, "trend": "hot"},
        {"team": "Astros", "elo_change_last_30_days": -15, "trend": "cold"},
    ],
    "2025-06": [
        {"team": "Dodgers", "elo_change_last_30_days": 35, "trend": "hot"},
        {"team": "Giants", "elo_change_last_30_days": -20, "trend": "cold"},
        {"team": "Yankees", "elo_change_last_30_days": 50, "trend": "hot"},
        {"team": "Astros", "elo_change_last_30_days": -25, "trend": "cold"},
    ],
}


def _normalize_year(year: str | int) -> str:
    return str(year)


def _find_team_record(team_name: str, year: str | int) -> dict | None:
    year_key = _normalize_year(year)
    teams_for_year = MOCK_PREDICTION_DB.get(year_key)
    if not teams_for_year:
        return None

    # Case-insensitive team matching
    for team, record in teams_for_year.items():
        if team.lower() == team_name.lower():
            return {"team": team, "record": record}
    return None


def _chance_response(metric_key: str, team_name: str, year: str | int) -> dict:
    year_key = _normalize_year(year)
    found = _find_team_record(team_name, year_key)
    if not found:
        return {
            "error": "not_found",
            "message": f"Sorry, I don't have prediction information for team '{team_name}' in year '{year_key}'.",
        }

    team = found["team"]
    record = found["record"]
    return {
        "team": team,
        "year": year_key,
        metric_key: record[metric_key],  # decimal percentage (0.0 - 1.0)
    }


def get_make_playoffs_chance(team_name: str, year: str | int) -> dict:
    """Get a team's chance of making the playoffs for a year.
    Args:
        team_name (str): Team name.
        year (str | int): Year in YYYY format.
    Returns:
        dict: Includes 'team', 'year', and 'make_playoffs_chance' (decimal percentage).
    """
    print(f"--- Tool: get_make_playoffs_chance called for team: {team_name}, year: {year} ---")
    return _chance_response("make_playoffs_chance", team_name, year)


def get_win_division_chance(team_name: str, year: str | int) -> dict:
    """Get a team's chance of winning its division for a year.
    Args:
        team_name (str): Team name.
        year (str | int): Year in YYYY format.
    Returns:
        dict: Includes 'team', 'year', and 'win_division_chance' (decimal percentage).
    """
    print(f"--- Tool: get_win_division_chance called for team: {team_name}, year: {year} ---")
    return _chance_response("win_division_chance", team_name, year)


def get_win_world_series_chance(team_name: str, year: str | int) -> dict:
    """Get a team's chance of winning the World Series for a year.
    Args:
        team_name (str): Team name.
        year (str | int): Year in YYYY format.
    Returns:
        dict: Includes 'team', 'year', and 'win_world_series_chance' (decimal percentage).
    """
    print(f"--- Tool: get_win_world_series_chance called for team: {team_name}, year: {year} ---")
    return _chance_response("win_world_series_chance", team_name, year)


# TODO: make less generic?
def get_prediction_for_team_given_a_year(team_name: str, year: str | int) -> dict:
    """Get a team's generic prediction chance for a year.
    Args:
        team_name (str): Team name.
        year (str | int): Year in YYYY format.
    Returns:
        dict: Includes 'team', 'year', and 'prediction_for_team_given_a_year' (decimal percentage).
    """
    print(f"--- Tool: get_prediction_for_team_given_a_year called for team: {team_name}, year: {year} ---")
    return _chance_response("prediction_for_team_given_a_year", team_name, year)


def get_elo_score(team_name: str, year: str | int) -> dict:
    """Get a team's ELO score for a given year.
    Args:
        team_name (str): Team name.
        year (str | int): Year in YYYY format.
    Returns:
        dict: Includes 'team', 'year', and 'elo_score'.
    """
    print(f"--- Tool: get_elo_score called for team: {team_name}, year: {year} ---")
    year_key = _normalize_year(year)
    teams_for_year = MOCK_ELO_DB.get(year_key)
    if not teams_for_year:
        return {
            "error": "not_found",
            "message": f"Sorry, I don't have ELO data for year '{year_key}'.",
        }
    
    for team, elo in teams_for_year.items():
        if team.lower() == team_name.lower():
            return {"team": team, "year": year_key, "elo_score": elo}
    
    return {
        "error": "not_found",
        "message": f"Sorry, I don't have ELO data for team '{team_name}' in year '{year_key}'.",
    }


def get_elo_leaderboard(year: str | int, top_n: int = 30) -> dict:
    """Get a sorted ELO leaderboard for all MLB teams in a season.
    Args:
        year (str | int): Year in YYYY format.
        top_n (int): Number of top teams to return (default: 30).
    Returns:
        dict: Includes 'year' and 'leaderboard' list of {rank, team, elo_score}.
    """
    print(f"--- Tool: get_elo_leaderboard called for year: {year}, top_n: {top_n} ---")
    year_key = _normalize_year(year)
    teams_for_year = MOCK_ELO_DB.get(year_key)
    if not teams_for_year:
        return {
            "error": "not_found",
            "message": f"Sorry, I don't have ELO data for year '{year_key}'.",
        }
    
    sorted_teams = sorted(teams_for_year.items(), key=lambda x: x[1], reverse=True)
    leaderboard = [
        {"rank": i + 1, "team": team, "elo_score": elo}
        for i, (team, elo) in enumerate(sorted_teams[:top_n])
    ]
    
    return {"year": year_key, "leaderboard": leaderboard}


def get_elo_trend(team_name: str, start_year: str | int, end_year: str | int) -> dict:
    """Track ELO changes for a team across multiple seasons.
    Args:
        team_name (str): Team name.
        start_year (str | int): Start year in YYYY format.
        end_year (str | int): End year in YYYY format.
    Returns:
        dict: Includes 'team', list of {year, elo_score}, and 'delta'.
    """
    print(f"--- Tool: get_elo_trend called for team: {team_name}, start: {start_year}, end: {end_year} ---")
    start_key = _normalize_year(start_year)
    end_key = _normalize_year(end_year)
    
    trend_data = []
    start_elo = None
    end_elo = None
    
    for year in range(int(start_key), int(end_key) + 1):
        year_str = str(year)
        year_elos = MOCK_ELO_DB.get(year_str, {})
        
        for team, elo in year_elos.items():
            if team.lower() == team_name.lower():
                trend_data.append({"year": year_str, "elo_score": elo})
                if start_elo is None:
                    start_elo = elo
                end_elo = elo
    
    if not trend_data:
        return {
            "error": "not_found",
            "message": f"Sorry, I don't have ELO trend data for team '{team_name}' between '{start_key}' and '{end_key}'.",
        }
    
    delta = end_elo - start_elo if (start_elo and end_elo) else 0
    return {"team": team_name, "trend_data": trend_data, "delta": delta}


def get_matchup_win_probability(home_team: str, away_team: str, year: str | int) -> dict:
    """Predict head-to-head win probability using ELO ratings.
    Args:
        home_team (str): Home team name.
        away_team (str): Away team name.
        year (str | int): Year in YYYY format.
    Returns:
        dict: Includes {home_win_prob, away_win_prob, elo_diff}.
    """
    print(f"--- Tool: get_matchup_win_probability called for {home_team} vs {away_team}, year: {year} ---")
    year_key = _normalize_year(year)
    teams_for_year = MOCK_ELO_DB.get(year_key, {})
    
    home_elo = None
    away_elo = None
    
    for team, elo in teams_for_year.items():
        if team.lower() == home_team.lower():
            home_elo = elo
        if team.lower() == away_team.lower():
            away_elo = elo
    
    if not home_elo or not away_elo:
        return {
            "error": "not_found",
            "message": f"Sorry, I don't have ELO data for one or both teams in year '{year_key}'.",
        }
    
    elo_diff = home_elo - away_elo
    home_win_prob = 50 + (elo_diff / 32)
    away_win_prob = 100 - home_win_prob
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "year": year_key,
        "home_win_prob": round(home_win_prob, 2),
        "away_win_prob": round(away_win_prob, 2),
        "elo_diff": elo_diff,
    }


def get_peak_elo(team_name: str) -> dict:
    """Find a team's highest or lowest ELO rating in history.
    Args:
        team_name (str): Team name.
    Returns:
        dict: Includes {peak_elo, peak_year, lowest_elo, lowest_year}.
    """
    print(f"--- Tool: get_peak_elo called for team: {team_name} ---")
    elo_history = []
    
    for year, teams in MOCK_ELO_DB.items():
        for team, elo in teams.items():
            if team.lower() == team_name.lower():
                elo_history.append({"year": year, "elo": elo})
    
    if not elo_history:
        return {
            "error": "not_found",
            "message": f"Sorry, I don't have ELO history for team '{team_name}'.",
        }
    
    peak = max(elo_history, key=lambda x: x["elo"])
    lowest = min(elo_history, key=lambda x: x["elo"])
    
    return {
        "team": team_name,
        "peak_elo": peak["elo"],
        "peak_year": peak["year"],
        "lowest_elo": lowest["elo"],
        "lowest_year": lowest["year"],
    }


def get_world_series_elo_history(year_range_start: str | int, year_range_end: str | int) -> dict:
    """Analyze ELO ratings of World Series winners and losers historically.
    Args:
        year_range_start (str | int): Start year in YYYY format.
        year_range_end (str | int): End year in YYYY format.
    Returns:
        dict: Includes list of {year, winner, winner_elo, loser, loser_elo}.
    """
    print(f"--- Tool: get_world_series_elo_history called for range: {year_range_start} - {year_range_end} ---")
    start_key = _normalize_year(year_range_start)
    end_key = _normalize_year(year_range_end)
    
    history = []
    for year in range(int(start_key), int(end_key) + 1):
        year_str = str(year)
        ws_entry = MOCK_WORLD_SERIES_DB.get(year_str)
        year_elos = MOCK_ELO_DB.get(year_str, {})
        
        if ws_entry and year_elos:
            winner = ws_entry["winner"]
            loser = ws_entry["loser"]
            history.append({
                "year": year_str,
                "winner": winner,
                "winner_elo": year_elos.get(winner),
                "loser": loser,
                "loser_elo": year_elos.get(loser),
            })
    
    return {"year_range": f"{start_key}-{end_key}", "history": history}


def get_elo_before_after_deadline(team_name: str, year: str | int) -> dict:
    """Measure ELO changes before and after the trade deadline.
    Args:
        team_name (str): Team name.
        year (str | int): Year in YYYY format.
    Returns:
        dict: Includes {pre_deadline_elo, post_deadline_elo, elo_change}.
    """
    print(f"--- Tool: get_elo_before_after_deadline called for team: {team_name}, year: {year} ---")
    year_key = _normalize_year(year)
    deadline_teams = MOCK_TRADE_DEADLINE_DB.get(year_key, {})
    
    deadline_entry = None
    for team, data in deadline_teams.items():
        if team.lower() == team_name.lower():
            deadline_entry = data
            break
    
    if not deadline_entry:
        return {
            "error": "not_found",
            "message": f"Sorry, I don't have trade deadline ELO data for team '{team_name}' in year '{year_key}'.",
        }
    
    elo_change = deadline_entry["post"] - deadline_entry["pre"]
    return {
        "team": team_name,
        "year": year_key,
        "pre_deadline_elo": deadline_entry["pre"],
        "post_deadline_elo": deadline_entry["post"],
        "elo_change": elo_change,
    }


def get_hot_cold_teams(year: str | int, month: str) -> dict:
    """Identify teams on the biggest ELO winning or losing streaks.
    Args:
        year (str | int): Year in YYYY format.
        month (str): Month (e.g., "06" for June).
    Returns:
        dict: List of {team, elo_change_last_30_days, trend}.
    """
    print(f"--- Tool: get_hot_cold_teams called for year: {year}, month: {month} ---")
    year_key = _normalize_year(year)
    month_key = f"{year_key}-{month}"
    teams_data = MOCK_HOT_COLD_DB.get(month_key)
    
    if not teams_data:
        return {
            "error": "not_found",
            "message": f"Sorry, I don't have hot/cold team data for '{month_key}'.",
        }
    
    return {"year": year_key, "month": month, "teams": teams_data}
