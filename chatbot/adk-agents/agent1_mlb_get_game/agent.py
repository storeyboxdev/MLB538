from google.adk.agents import Agent

MODEL = "gemini-2.5-flash"

from .tools import get_games_on_date

root_agent = Agent(
    name="get_matches_on_date_agent_v1",
    model=MODEL,
    description="Provides game information for specific dates.",
    instruction="You are a helpful MLB assistant. "
                "When the user asks for the games on a date, "
                "use the 'get_games_on_date' tool to find the information. "
                "If the tool returns an error, inform the user politely. "
                "If the tool is successful, present the data clearly.",
    tools=[get_games_on_date],
)
