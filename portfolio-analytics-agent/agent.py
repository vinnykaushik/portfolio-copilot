import os
from google.adk import Agent
from mcp import StdioServerParameters
from pydantic import BaseModel
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

TARGET_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mcp_portfolio_analytics.py"
)

portfolio_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="python",
        args=[
            os.path.abspath(TARGET_FOLDER_PATH),
        ],
    )
)


root_agent = Agent(
    name="portfolio-analytics-agent",
    description="Agent for portfolio analytics",
    tools=[portfolio_toolset],
)
