def get_games_on_date(date: str) -> dict:
    """Retrieves the game information of games that played on a given date. 
    Args:
        date (str): The date of a game in YYYY-MM-DD format.
    Returns:
        dict: A dictionary containing:
              - 'home_team' (str): The home team that played
              - 'away_team' (str): The away team that played
              - 'home_team_score' (int): The home team's score
              - 'away_team_score' (int): The away team's score
    """
    print(f"--- Tool: get_games_on_date called for date: {date} ---")

    # Mock game data
    mock_game_db = {
        "2022-11-16":
            {
                "home_team": "Dodgers", 
                "away_team": "Giants",
                "home_team_score": 5,
                "away_team_score": 10,
            },
    }

    if date in mock_game_db:
        return {"games": mock_game_db[date]}
    else:
        return {"error": "not_found", "message": f"Sorry, I don't have game information for '{date}'."}
