from google.adk.agents import Agent
from .tools import (
    get_make_playoffs_chance,
    get_win_division_chance,
    get_win_world_series_chance,
    get_prediction_for_team_given_a_year,
)

MODEL = "gemini-2.5-flash"

root_agent = Agent(
    name="mlb_predictor_agent_v1",
    model=MODEL,
    description="The main coordinator agent. Handles prediction requests for Major League Baseball (MLB).",
    instruction="""You are the main MLB predictor agent. Your primary responsibility is to get predictions about MLB.

    If user inputs a number for year, convert to a string for the function.

    Use get_make_playoffs_chance ONLY for the chance (decimal that you convert to percent) that a team makes the playoffs in a given year.

    Use get_win_division_chance ONLY for the chance (decimal that you convert to percent) that a team wins their division in a given year.

    Use get_win_world_series_chance ONLY for the chance (decimal that you convert to percent) that a team wins the World Series in a given year.

    For other MLB team prediction-of-a-year requests, use ONLY get_prediction_for_team_given_a_year.

    For anything else, respond appropriately or state you cannot handle it.""",
    tools=[
        get_make_playoffs_chance,
        get_win_division_chance,
        get_win_world_series_chance,
        get_prediction_for_team_given_a_year,
    ],
)
