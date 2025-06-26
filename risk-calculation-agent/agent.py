import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from typing import List, Any

# Initialize at module level (keeping your existing approach)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is not None:
    os.environ["GOOGLE_API_KEY"] = google_api_key

MPC_SERVER_LOCATION = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "mcp_risk_calculation.py"
)
client = MultiServerMCPClient(
    {
        "risk-calculation-agent": {
            "command": "fastmcp",
            "args": ["run", MPC_SERVER_LOCATION],
            "transport": "stdio",
        },  # type: ignore
    }
)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": 1}}


async def get_tools():
    """Get tools from the MCP client."""
    return await client.get_tools()


def format_mcp_tools(tools_list: List) -> str:
    """
    Format an MCP tool list into human-readable markdown format.

    Args:
        tools_list: List of StructuredTool objects

    Returns:
        str: Formatted markdown string
    """
    if not tools_list:
        return "No tools available.\n"

    markdown_output = []

    for tool in tools_list:
        # Tool name as heading 2
        markdown_output.append(f"## {tool.name}")
        markdown_output.append("")

        # Description
        if hasattr(tool, "description") and tool.description:
            # Clean up description - remove extra whitespace and format nicely
            description = tool.description.strip()
            # Replace multiple newlines with double newlines for proper markdown
            description = "\n\n".join(
                line.strip() for line in description.split("\n") if line.strip()
            )
            markdown_output.append(f"**Description:** {description}")
            markdown_output.append("")

        # Schema information
        if hasattr(tool, "args_schema") and tool.args_schema:
            markdown_output.append("**Parameters:**")
            markdown_output.append("")

            schema = tool.args_schema

            # Handle required parameters
            if "required" in schema:
                required_params = schema["required"]
            else:
                required_params = []

            # Process properties
            if "properties" in schema:
                for param_name, param_info in schema["properties"].items():
                    is_required = param_name in required_params
                    required_text = " *(required)*" if is_required else " *(optional)*"

                    # Get parameter type
                    param_type = param_info.get("type", "unknown")

                    # Handle references to definitions
                    if "$ref" in param_info:
                        ref_path = param_info["$ref"]
                        if ref_path.startswith("#/$defs/"):
                            param_type = ref_path.split("/")[-1]

                    markdown_output.append(
                        f"- `{param_name}` ({param_type}){required_text}"
                    )

                    # Add description if available
                    if "title" in param_info:
                        markdown_output.append(f"  - {param_info['title']}")

                    markdown_output.append("")

            # Handle schema definitions
            if "$defs" in schema:
                markdown_output.append("**Schema Definitions:**")
                markdown_output.append("")

                for def_name, def_info in schema["$defs"].items():
                    markdown_output.append(f"### {def_name}")

                    if "description" in def_info:
                        markdown_output.append(f"{def_info['description']}")
                        markdown_output.append("")

                    if "properties" in def_info:
                        markdown_output.append("**Properties:**")

                        required_fields = def_info.get("required", [])

                        for prop_name, prop_info in def_info["properties"].items():
                            is_req = prop_name in required_fields
                            req_text = " *(required)*" if is_req else " *(optional)*"
                            prop_type = prop_info.get("type", "unknown")

                            # Handle array types
                            if prop_type == "array" and "items" in prop_info:
                                items_info = prop_info["items"]
                                if "$ref" in items_info:
                                    item_type = items_info["$ref"].split("/")[-1]
                                    prop_type = f"array[{item_type}]"
                                else:
                                    item_type = items_info.get("type", "unknown")
                                    prop_type = f"array[{item_type}]"

                            markdown_output.append(
                                f"- `{prop_name}` ({prop_type}){req_text}"
                            )

                            if "description" in prop_info:
                                markdown_output.append(
                                    f"  - {prop_info['description']}"
                                )

                        markdown_output.append("")

        # Response format
        if hasattr(tool, "response_format") and tool.response_format:
            markdown_output.append(f"**Response Format:** {tool.response_format}")
            markdown_output.append("")

        # Add separator between tools (except for the last one)
        if tool != tools_list[-1]:
            markdown_output.append("---")
            markdown_output.append("")

    return "\n".join(markdown_output)


# Initialize tools and agent at module level
tools = asyncio.run(get_tools())
tool_descriptions = format_mcp_tools(tools)

risk_calculation_prompt = f"""
# Overview - Risk Calculation Agent
You are a helpful risk calculation agent. Your job is to analyze financial portfolios and calculate various risk metrics based on the provided data.
Your capabilities are limited to the tools available to you, which are listed below:

# Tools
{tool_descriptions}

# Instructions
- Use the tools to perform calculations and return results in a structured format.
- Format the input for tools according to the schema definitions provided in the tool descriptions. Always ensure that the input matches the expected schema.
- When presenting results, format them clearly and explain what each metric means.
- If asked about portfolio analysis, break down the results by investment type and provide insights.
"""

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

agent = create_react_agent(
    name="Risk Calculation Agent",
    model=llm,
    prompt=risk_calculation_prompt,
    tools=tools,
    checkpointer=checkpointer,
)


def extract_response_content(response: Any) -> str:
    """Extract the actual response content from the agent response."""
    try:
        # Check if response is a dictionary with messages
        if isinstance(response, dict) and "messages" in response:
            agent_messages = response["messages"]
            if agent_messages:
                last_message = agent_messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                else:
                    return str(last_message)
            else:
                return "No response generated."

        # If response has messages attribute
        elif hasattr(response, "messages"):
            agent_messages = response.messages  # type: ignore
            if agent_messages:
                last_message = agent_messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                else:
                    return str(last_message)
            else:
                return "No response generated."

        # If response has content attribute directly
        elif hasattr(response, "content"):
            return response.content  # type: ignore

        # Fallback to string representation
        else:
            return str(response)

    except Exception as e:
        return f"Error extracting response: {str(e)}"


async def get_agent_response(message: str) -> Any:
    """Get response from the agent."""
    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]}, config=config  # type: ignore
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"


async def main():
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": """what is the conditional VaR of this portfolio?'
                                  {
                                    "customer_id": "33333",
                                    "customer_name": "Carl Pei",
                                    "portfolio_value": 320000,
                                    "investments": [
                                        {
                                            "type": "stocks",
                                            "symbol": "NVDA",
                                            "quantity": 40,
                                        },
                                        {
                                            "type": "stocks",
                                            "symbol": "AMD",
                                            "quantity": 18,
                                        },
                                        {
                                            "type": "mutual_funds",
                                            "name": "Technology Select Sector SPDR Fund",
                                            "value": 90000,
                                        },
                                        {
                                            "type": "stocks",
                                            "symbol": "INTC",
                                            "quantity": 25,
                                        },
                                    ],
                                    "last_updated": "2025-06-18",
                                }""",
                }
            ]
        }
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
