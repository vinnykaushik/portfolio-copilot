from abc import get_cache_token
import os
import asyncio
from dotenv import load_dotenv
from httpx import get
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from typing import AsyncIterable, List, Any, Literal
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize at module level (keeping your existing approach)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is not None:
    os.environ["GOOGLE_API_KEY"] = google_api_key

MPC_SERVER_LOCATION = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "mcp_risk_calculation.py"
)

# Pre-initialize the MCP client and tools at module level
logger.info("Initializing MCP client...")
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

# Pre-initialize tools and agent
logger.info("Initializing tools and agent...")
tools = asyncio.run(client.get_tools())  # Get tools once at startup


def format_mcp_tools(tools_list: List) -> str:
    # ... (keep your existing format_mcp_tools function) ...
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


def get_mcp_tools() -> List:
    """
    Retrieve the list of tools from the MCP client.
    """
    return tools  # Return pre-initialized tools


# Initialize agent at module level
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
- If needed, reformat the input data to match the tool's schema.
- When presenting results, format them clearly and explain what each metric means.
- If asked about portfolio analysis, break down the results by investment type and provide insights.
"""

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class RiskCalculationAgent:
    """Risk Calculation Agent Example."""

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more information to complete the request."
        "Set response status to error if there is an error while processing the request."
        "Set response status to completed if the request is complete."
    )

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = tools  # Use pre-initialized tools
        logger.info(f"Creating Risk Calculation Agent graph with tools {self.tools}")

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=checkpointer,
            prompt=risk_calculation_prompt,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )
        logger.info("Risk Calculation Agent initialized!")

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        logger.info(f"Streaming query: {query}")
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": context_id}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):  # type: ignore
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing query with tools...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing tool call...",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        logger.info(f"Getting agent response for config: {config}")
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get("structured_response")
        logger.info(f"Structured response: {structured_response}")
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "We are unable to process your request at the moment. "
                "Please try again."
            ),
        }


agent_instance = RiskCalculationAgent()


def initialize_agent():
    """Return the pre-initialized agent and config."""
    return agent_instance, config


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
        response = await agent_instance.get_agent_response(
            {"messages": [{"role": "user", "content": message}]}, config=config  # type: ignore
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"
