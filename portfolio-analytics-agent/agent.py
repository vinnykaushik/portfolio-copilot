import os
import sys
import asyncio
from google.adk import Agent
from mcp import StdioServerParameters, stdio_client, ClientSession
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

TARGET_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mcp_portfolio_analytics.py"
)

LLM_MODEL = "gemini-2.0-flash"

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


async def get_tool_descriptions():
    """Get tool descriptions directly from MCP server."""
    server_params = StdioServerParameters(
        command="python", args=[os.path.abspath(TARGET_FOLDER_PATH)]
    )

    try:
        async with stdio_client(server_params) as (stdio, write):
            async with ClientSession(stdio, write) as session:
                await session.initialize()
                response = await session.list_tools()
                return response.tools
    except Exception as e:
        print(f"Error connecting to MCP server: {e}")
        return []


def format_tool_descriptions(tools):
    """Format tools into a readable description."""
    if not tools:
        return "No tools available in the portfolio toolset."

    descriptions = []

    for tool in tools:
        tool_desc = f"## {tool.name}\n"
        tool_desc += f"**Description:** {tool.description}\n\n"

        # Add input schema information if available
        if hasattr(tool, "inputSchema") and tool.inputSchema:
            tool_desc += "**Parameters:**\n"
            schema = tool.inputSchema

            if "properties" in schema:
                for param_name, param_info in schema["properties"].items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get(
                        "description", "No description available"
                    )
                    required = param_name in schema.get("required", [])
                    required_text = " (required)" if required else " (optional)"

                    tool_desc += f"- `{param_name}` ({param_type}){required_text}: {param_desc}\n"

            tool_desc += "\n"

        descriptions.append(tool_desc)

    return "\n".join(descriptions)


def get_toolset_description():
    """Get toolset description, handling event loop properly."""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in a running loop, so we can't use asyncio.run()
        # Instead, we'll provide a fallback description
        return """
## calculate_position_concentration
**Description:** Calculates position-level concentration metrics using live Yahoo Finance prices. Returns the portfolio value, Herfindahl-Hirschman Index (HHI), top-N concentration, and normalized position weights.

**Parameters:**
- `portfolio` (CustomerPortfolio) (required): Customer portfolio containing ID, name, and list of investments
- `top_n` (int) (optional): Number of top holdings to include in the concentration metric

## analyze_portfolio_exposures
**Description:** Analyzes portfolio exposures across sector, geography, and market cap dimensions using live Yahoo Finance data. Returns percentage allocations and dollar values for each category.

**Parameters:**
- `portfolio` (CustomerPortfolio) (required): Customer portfolio containing ID, name, and list of investments
- `min_threshold` (float) (optional): Minimum weight threshold to include in results (0.01 = 1%)
"""
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        try:
            tools = asyncio.run(get_tool_descriptions())
            return format_tool_descriptions(tools)
        except Exception as e:
            print(f"Failed to get tool descriptions: {e}")
            return "Failed to load tool descriptions from MCP server."


# Get the toolset description
toolset_description = get_toolset_description()

portfolio_analytics_prompt = f"""
# Overview
You are a helpful portfolio analytics agent that provides insights into customer portfolios.
Your capabilities include calculating position concentration, analyzing portfolio exposure, and comparing multiple portfolios on these metrics.

# Available Tools
{toolset_description}

# Usage Guidelines
When analyzing portfolios:

1. **Position Concentration Analysis**: Use `calculate_position_concentration` to understand how concentrated a portfolio is across its holdings. The HHI (Herfindahl-Hirschman Index) ranges from 0 (perfectly diversified) to 1 (completely concentrated).

2. **Exposure Analysis**: Use `analyze_portfolio_exposures` to break down portfolio allocations by:
   - **Sector**: Industry classifications (Technology, Healthcare, etc.)
   - **Geography**: Country-based exposure (US, International, etc.)  
   - **Market Cap**: Small Cap (<$2B), Mid Cap ($2B-$10B), Large Cap (>$10B)

3. **Data Requirements**: Portfolio data should include:
   - Customer identification (ID and name)
   - Investment details with proper types ("stocks", "crypto", "mutual_funds")
   - For stocks/crypto: symbol and quantity
   - For mutual funds: name and current value
   - Last updated timestamp

4. **Error Handling**: If tools encounter errors (e.g., invalid symbols, network issues), explain the issues clearly and suggest alternatives or data corrections.

5. **Interpretation**: Always provide context for the metrics you calculate:
   - Explain what concentration levels mean for risk
   - Highlight notable exposures or concentrations
   - Suggest diversification improvements when appropriate

Remember to validate portfolio data structure before calling tools and provide meaningful insights rather than just raw numbers.
"""

root_agent = Agent(
    name="portfolio_analytics_agent",
    description="Agent for portfolio analytics with comprehensive toolset integration",
    model=LLM_MODEL,
    instruction=portfolio_analytics_prompt,
    tools=[portfolio_toolset],
)

# Optional: Print the generated tool descriptions for debugging
if __name__ == "__main__":
    print("Generated Tool Descriptions:")
    print("=" * 50)
    print(toolset_description)
    print("\n" + "=" * 50)
    print("Full Agent Prompt:")
    print(portfolio_analytics_prompt)
