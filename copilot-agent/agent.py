from json import tool
import json
from types import CoroutineType
from dotenv import load_dotenv
from uuid import uuid4
import os
import langfuse
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from typing import Callable, Dict, Literal, Any
import logging
import httpx
from a2a.client.client import A2AClient, A2ACardResolver
from a2a.types import (
    SendMessageResponse,
    GetTaskResponse,
    SendMessageSuccessResponse,
    Task,
    SendMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
)
import asyncio
from langchain_core.tools import tool

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
TRACKING_AUTH_USERNAME = os.getenv("TRACKING_AUTH_USERNAME", "")
TRACKING_AUTH_PASSWORD = os.getenv("TRACKING_AUTH_PASSWORD", "")

TIMEOUT_SETTINGS = httpx.Timeout(
    connect=30.0,  # 30 seconds to connect
    read=300.0,  # 5 minutes to read response
    write=30.0,  # 30 seconds to write
    pool=30.0,  # 30 seconds for pool operations
)

from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()


def _get_a2a_descriptions() -> str:
    map = {
        "customer_portfolio_tool": "http://localhost:8888",
        "risk_calculation_tool": "http://localhost:9999",
        "performance_tracking_tool": "https://performance-tracking-agent-906909573150.us-central1.run.app",
    }
    descriptions = ""
    for agent, base_url in map.items():
        # Add verify=False for HTTPS URLs
        if base_url.startswith("https"):
            agent_card = httpx.get(base_url + "/.well-known/agent.json", verify=False)
        else:
            agent_card = httpx.get(base_url + "/.well-known/agent.json")
        descriptions += f"- **{agent}**:\n" f"Description: {agent_card.json()}\n\n"
    return descriptions


def _create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a message."""
    payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": uuid4().hex,
        },
    }

    if task_id:
        payload["message"]["taskId"] = task_id

    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


def _print_json_response(response: Any, description: str) -> str:
    """Helper function to print and return the JSON representation of a response."""
    answer: str = f"--- {description} ---"
    print(answer)
    if hasattr(response, "root"):
        answer += f"{response.root.model_dump_json(exclude_none=True)}\n"
        print(f"{response.root.model_dump_json(exclude_none=True)}\n")
    else:
        answer += f"{response.model_dump(mode='json', exclude_none=True)}\n"
        print(f"{response.model_dump(mode='json', exclude_none=True)}\n")
    return answer


async def _send_message_to_a2a_agent(message: str, client: A2AClient) -> str | None:
    """Sends a message to an A2A agent and retrieves the response."""
    try:
        payload = _create_send_message_payload(message)
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**payload)
        )
        response: SendMessageResponse = await client.send_message(
            request,
            http_kwargs={
                "auth": (
                    TRACKING_AUTH_USERNAME,
                    TRACKING_AUTH_PASSWORD,
                )
            },
        )
        answer = _print_json_response(response, "📥 Single Turn Request Response")

        if not isinstance(response.root, SendMessageSuccessResponse):
            print("received non-success response. Aborting get task ")
            return f"Received non-success response from agent: {answer}"

        if not isinstance(response.root.result, Task):
            print("received non-task response. Aborting get task ")
            return f"Received non-task response from agent {answer}"

        task_id: str = response.root.result.id
        print(f"--- ❔ Query Task (ID: {task_id}) ---")

        # Query the task with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                get_request = GetTaskRequest(
                    id=str(uuid4()), params=TaskQueryParams(id=task_id)
                )
                get_response: GetTaskResponse = await client.get_task(get_request)
                answer = _print_json_response(get_response, "📥 Query Task Response")

                return answer

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Attempt {attempt + 1} failed, retrying... ({e})")
                    await asyncio.sleep(2)
                else:
                    raise

    except Exception as e:
        logger.error(f"Error in _send_message_to_a2a_agent: {e}")
        return f"Error processing request: {str(e)}"


# Define the response format for the agent. Ensures we can see the status of the agent's response.
class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    thought: str = ""
    message: str


class PortfolioCopilotAgent:

    SYSTEM_PROMPT = f"""
    # Overview
    You are a helpful portfolio copilot agent. You use your tools to assist users with their portfolio management tasks. You fetch portfolio data and perform various operations on said data.
    
    # Tools
    Note that the tools that you have access to are themselves agents, and you may need to converse with them to complete your task.
    You have access to the following tools:
    {_get_a2a_descriptions()}

    # Instructions
    - Always follow the input schema requested by each tool. If needed, format the input before passing it to the tool.
    - Please be concise and clear in your responses.
    - If you need more information from the user, ask for it clearly.
    - Make sure to return results provided by the tools to the user.

    # Examples
    Query: "Can you find me Carl Pei's portfolio and provide concentration risk analysis?"
    Response: "Sure! Here's Carl Pei's portfolio:
    - Stocks: 40 shares of NVDA, 18 shares of AMD, and 25 shares of INTC
    - Mutual Funds: Technology Select Sector SPDR Fund valued at $90,000
    - Bonds: Corporate Bond Index Fund valued at $45,000
    The total portfolio value is $320,000, as of 2025-06-18.

    And here's the concentration analysis:
        Portfolio Value: $10,828.14
        Herfindahl-Hirschman Index (HHI): 0.5251. The HHI ranges from close to 0 to 1, with higher values indicating greater concentration. A value of 0.5251 suggests moderate concentration.
        Top-N Concentration: 1.0, meaning that the top holdings (in this case, all holdings since we only have three stocks) account for 100% of the portfolio value.
        Normalized Position Weights:
        NVDA: 66.07%
        AMD: 29.41%
        INTC: 4.52%
    This analysis reveals that the portfolio is heavily weighted towards NVDA, which makes up the majority of the portfolio's value. AMD represents a significant portion as well, while INTC has a relatively small position. The HHI indicates moderate concentration, which is also reflected in the top-N concentration and position weights.

    # CRITICAL INSTRUCTIONS FOR REPORTING:
    1. **ALWAYS include the FULL response from tools in your answer to the user**
    2. **DO NOT summarize or condense the information received from tools**
    3. **When a tool returns calculations, data, or analysis, reproduce it entirely in your response**
    4. **Your role is to orchestrate and relay information, not to summarize it**

    # Example of INCORRECT behavior:
    User: "Calculate the VaR for my portfolio"
    Tool returns: [detailed VaR calculation with numbers]
    WRONG Response: "I have calculated the VaR for your portfolio. Please see the details in the message."

    # Example of CORRECT behavior:
    User: "Calculate the VaR for my portfolio"
    Tool returns: [detailed VaR calculation with numbers]
    CORRECT Response: "I've calculated the VaR for your portfolio. Here are the results:
    [Include ALL the details from the tool response]"

    # Instructions
    - Pass through ALL information received from tools
    - Format the information nicely for readability, but don't omit anything
    - Add brief contextual explanations only if needed, but always include the full data
    """

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more information to complete the request."
        "Set response status to error if there is an error while processing the request."
        "Set response status to completed if the request is complete."
        "Record what you are thinking and next steps in the thought field."
    )

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.agent_configs = {
            "customer_portfolio_tool": {
                "url": "http://localhost:8888",
                "auth_required": False,
            },
            "risk_calculation_tool": {
                "url": "http://localhost:9999",
                "auth_required": False,
            },
            "performance_tracking_tool": {
                "url": "https://performance-tracking-agent-906909573150.us-central1.run.app",
                "auth_required": True,
            },
        }

        # Dynamically create tools
        self.tools = self._create_tools_dynamically()

        logger.info("🏃‍➡️ Creating Portfolio Copilot Agent...")
        logger.info(f"Using tools: {[tool for tool in self.tools]}")

        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            name="portfolio_copilot_agent",
            prompt=self.SYSTEM_PROMPT,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
            checkpointer=InMemorySaver(),
        )

    def _create_tools_dynamically(self) -> list:
        """Dynamically create tool functions based on agent configurations."""
        tools = []

        for agent_name, config in self.agent_configs.items():
            # Fetch agent card
            agent_card = self._fetch_agent_card(config["url"])

            # Create tool function
            tool_func = self._create_tool_function(
                agent_name=agent_name,
                base_url=config["url"],
                agent_card=agent_card,
                auth_required=config["auth_required"],
            )

            tools.append(tool_func)

        return tools

    def _fetch_agent_card(self, base_url: str) -> dict:
        """Fetch agent card from the given URL."""
        try:
            if base_url.startswith("https"):
                response = httpx.get(base_url + "/.well-known/agent.json", verify=False)
            else:
                response = httpx.get(base_url + "/.well-known/agent.json")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch agent card from {base_url}: {e}")
            return {}

    def _create_tool_function(
        self, agent_name: str, base_url: str, agent_card: dict, auth_required: bool
    ) -> Callable:
        """Create a tool function with proper description from agent card."""

        # Build comprehensive description
        skills_desc = []
        examples = []

        for skill in agent_card.get("skills", []):
            skills_desc.append(f"- {skill['name']}: {skill['description']}")
            examples.extend(skill.get("examples", []))

        skills_text = "\n".join(skills_desc)
        examples_text = ", ".join(
            examples[:5]
        )  # Limit examples to avoid too long descriptions

        tool_description = f"""{agent_card.get('description', 'Agent for ' + agent_name)}

Available capabilities:
{skills_text}

Example queries: {examples_text}"""

        # Create the actual tool function
        async def tool_function(message: str) -> str:
            """This docstring will be replaced dynamically."""
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT_SETTINGS) as httpx_client:
                    http_kwargs = {}
                    if auth_required:
                        http_kwargs["auth"] = (
                            TRACKING_AUTH_USERNAME,
                            TRACKING_AUTH_PASSWORD,
                        )

                    client = await A2AClient.get_client_from_agent_card_url(
                        httpx_client,
                        base_url,
                        http_kwargs=http_kwargs if http_kwargs else None,
                    )
                    logger.info(f"✅ Connected to {agent_name}")
                    return (
                        await _send_message_to_a2a_agent(message, client)
                        or "No response"
                    )
            except Exception as e:
                logger.error(f"Error connecting to {agent_name}: {e}")
                return f"Error connecting to {agent_name}."

        # Set function metadata
        tool_function.__name__ = agent_name
        tool_function.__doc__ = tool_description

        # Apply the @tool decorator
        decorated_tool = tool(tool_function)

        return decorated_tool

    async def ainvoke_with_tracing(self, messages, config) -> dict[str, Any]:
        """Get response from the agent synchronously"""
        config["callbacks"] = [langfuse_handler]
        logger.info(f"config: {config}")
        return await self.agent.ainvoke(
            messages,
            config=config,
        )


if __name__ == "__main__":
    root_agent = PortfolioCopilotAgent()
    print("Portfolio Copilot Agent is ready to assist you!")
