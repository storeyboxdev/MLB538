from google.adk.agents import Agent
from .tools import (
    get_bigquery_mcp_toolset,
    get_health,
    forecast,
)

MODEL = "gemini-2.5-flash"
PROJECT_ID = "qwiklabs-asl-02-03bf2b8329ea"
DATASET = "mlb538"
bigquery_tools = get_bigquery_mcp_toolset()

bigquery_agent = Agent(
    name="bigquery_agent",
    model=MODEL,
    description="Runs analytics queries against BigQuery via MCP tools.",
    instruction=f"""You are a BigQuery analytics specialist.
    Default Google Cloud project is `{PROJECT_ID}`.
    Only query dataset `{DATASET}`.
    Use fully-qualified names like `{PROJECT_ID}.{DATASET}.<table_name>`.

    **Access Restrictions:**
    You only have access to the `{DATASET}` dataset. You do NOT have access 
    to other datasets. If a customer asks about admin 
    data, politely explain that you only have access to `{DATASET}`.    

    Use the available BigQuery MCP tools to:
    - inspect datasets/tables/schemas
    - run SQL queries
    - summarize query results clearly

    Prefer safe read-only queries unless the user explicitly asks otherwise.
    If required table or dataset is unknown, ask a clarifying question.""",
    tools=[bigquery_tools],
)

prediction_agent = Agent(
    name="prediction_agent",
    model=MODEL,
    description="Handles season forecasts specifically for the years 2024, 2025, and 2026.",
    instruction="""Use forecast for season simulation forecasts (season + optional sims).
    
    **Restrictions:**
    You can ONLY handle forecasts for the years 2024, 2025, and 2026. If a user requests a forecast for any other year, politely inform them that you only support 2024 through 2026.""",
    tools=[forecast],
)

health_agent = Agent(
    name="health_agent",
    model=MODEL,
    description="Checks MLB API availability and status.",
    instruction="""Use get_health to verify whether the MLB API is reachable and healthy.""",
    tools=[get_health],
)

root_agent = Agent(
    name="mlb_predictor_agent_v1",
    model=MODEL,
    description="Main MLB predictor coordinator.",
    instruction="""Route:
- API health/status checks -> health_agent
- season forecasts -> prediction_agent
- SQL/data exploration/BigQuery analytics -> bigquery_agent""",
    sub_agents=[
        health_agent,
        prediction_agent,
        bigquery_agent,
    ],
)
