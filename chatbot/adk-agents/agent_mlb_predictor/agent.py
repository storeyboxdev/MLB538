from google.adk.agents import Agent
from .tools import (
    get_make_playoffs_chance,
    get_win_division_chance,
    get_win_world_series_chance,
    get_prediction_for_team_given_a_year,
    get_elo_score,
    get_elo_leaderboard,
    get_elo_trend,
    get_matchup_win_probability,
    get_peak_elo,
    get_world_series_elo_history,
    get_elo_before_after_deadline,
    get_hot_cold_teams,
)

MODEL = "gemini-2.5-flash"

# Prediction subagent
prediction_agent = Agent(
    name="prediction_agent",
    model=MODEL,
    description="Handles playoff, division, and World Series predictions.",
    instruction="""You are a prediction specialist. Handle requests about:
    - Playoff chances for teams in a given year
    - Division win chances for teams in a given year
    - World Series win chances for teams in a given year
    - General team predictions for a given year
    
    If user inputs a number for year, convert to a string for the function.
    
    Use get_make_playoffs_chance ONLY for playoff prediction chances.
    Use get_win_division_chance ONLY for division win chances.
    Use get_win_world_series_chance ONLY for World Series win chances.
    Use get_prediction_for_team_given_a_year for other team predictions.
    
    Always convert decimal results to percentages for the user.""",
    tools=[
        get_make_playoffs_chance,
        get_win_division_chance,
        get_win_world_series_chance,
        get_prediction_for_team_given_a_year,
    ],
)

# ELO score subagent
elo_score_agent = Agent(
    name="elo_score_agent",
    model=MODEL,
    description="Gets ELO scores and rankings for MLB teams.",
    instruction="""You are an ELO specialist focused on scores and rankings. Handle requests for:
    - Individual team ELO scores for a specific year
    - ELO leaderboards and rankings for a season
    - Peak (highest and lowest) ELO ratings for teams
    
    If user inputs a number for year, convert to a string for the function.
    
    Use get_elo_score for a single team's ELO in a specific year.
    Use get_elo_leaderboard for ranking all teams by ELO in a year.
    Use get_peak_elo for a team's all-time high and low ELO ratings.
    
    Present results clearly with rankings and comparisons.""",
    tools=[
        get_elo_score,
        get_elo_leaderboard,
        get_peak_elo,
    ],
)

# ELO trend subagent
elo_trend_agent = Agent(
    name="elo_trend_agent",
    model=MODEL,
    description="Analyzes ELO changes and trends over time.",
    instruction="""You are an ELO trend analyst. Handle requests for:
    - ELO changes across multiple seasons for a team
    - Trade deadline impact on team ELO ratings
    - Hot and cold teams based on recent ELO changes
    
    If user inputs a number for year, convert to a string for the function.
    
    Use get_elo_trend for tracking ELO changes over time periods.
    Use get_elo_before_after_deadline for trade deadline analysis.
    Use get_hot_cold_teams to identify trending teams by month.
    
    Always highlight the overall delta and notable changes.""",
    tools=[
        get_elo_trend,
        get_elo_before_after_deadline,
        get_hot_cold_teams,
    ],
)

# Matchup and history subagent
matchup_history_agent = Agent(
    name="matchup_history_agent",
    model=MODEL,
    description="Predicts matchups and analyzes World Series history.",
    instruction="""You are a matchup and historical analyst. Handle requests for:
    - Head-to-head win probabilities between two teams
    - World Series historical ELO comparisons between winners and losers
    
    If user inputs a number for year, convert to a string for the function.
    
    Use get_matchup_win_probability for predicting game outcomes based on ELO.
    Use get_world_series_elo_history for analyzing World Series champion vs runner-up ELO ratings.
    
    Always present both teams' probabilities and explain the ELO differences.""",
    tools=[
        get_matchup_win_probability,
        get_world_series_elo_history,
    ],
)

# Root coordinator agent
root_agent = Agent(
    name="mlb_predictor_agent_v1",
    model=MODEL,
    description="The main coordinator agent. Handles prediction requests for Major League Baseball (MLB).",
    instruction="""You are the main MLB predictor agent. Your primary responsibility is to coordinate and delegate prediction requests about Major League Baseball.

    You have access to four specialized subagents:
    1. prediction_agent - handles playoff, division, and World Series predictions
    2. elo_score_agent - handles ELO scores, leaderboards, and peak ratings
    3. elo_trend_agent - handles ELO trends, trade deadline analysis, and hot/cold teams
    4. matchup_history_agent - handles head-to-head matchups and World Series history

    Route user requests to the appropriate subagent based on their query type:
    - For playoff/division/World Series/general predictions → prediction_agent
    - For ELO scores, leaderboards, or peak ratings → elo_score_agent
    - For ELO trends, deadline impact, or hot/cold teams → elo_trend_agent
    - For matchups or World Series history → matchup_history_agent

    If a request doesn't fit any category, respond appropriately or state you cannot handle it.""",
    sub_agents=[
        prediction_agent,
        elo_score_agent,
        elo_trend_agent,
        matchup_history_agent,
    ],
)
