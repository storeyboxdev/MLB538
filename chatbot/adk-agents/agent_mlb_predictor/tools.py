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
