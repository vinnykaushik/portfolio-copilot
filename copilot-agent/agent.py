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
from typing import Dict, Literal, Any
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
        "customer_portfolio_tool": "http://localhost:7777",
        "risk_calculation_tool": "http://localhost:9999",
        "portfolio_analytics_tool": "http://localhost:8888",
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


@tool
async def customer_portfolio_tool(message: str) -> str | None:
    """Sends a message to the customer portfolio agent and returns the response."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SETTINGS) as httpx_client:
            global customer_portfolio_client
            customer_portfolio_client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, "http://localhost:7777"
            )
            logger.info("✅ Connected to the customer portfolio agent.")
            return await _send_message_to_a2a_agent(message, customer_portfolio_client)
    except Exception as e:
        logger.error(f"Error connecting to the customer portfolio agent: {e}")
        return "Error connecting to the customer portfolio agent."


@tool
async def risk_calculation_tool(message: str) -> str | None:
    """Sends a message to the customer portfolio agent and returns the response."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SETTINGS) as httpx_client:
            global risk_calculation_agent
            risk_calculation_agent = await A2AClient.get_client_from_agent_card_url(
                httpx_client, "http://localhost:9999"
            )
            logger.info("✅ Connected to the risk calculation agent.")
            return await _send_message_to_a2a_agent(message, risk_calculation_agent)
    except Exception as e:
        logger.error(f"Error connecting to the risk calculation agent: {e}")
        return "Error connecting to the risk calculation agent."


@tool
async def portfolio_analytics_tool(message: str) -> str | None:
    """Sends a message to the portfolio analytics agent and returns the response."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SETTINGS) as httpx_client:
            global portfolio_analytics_agent
            portfolio_analytics_agent = await A2AClient.get_client_from_agent_card_url(
                httpx_client, "http://localhost:8888"
            )
            logger.info("✅ Connected to the portfolio analytics agent.")
            return await _send_message_to_a2a_agent(message, portfolio_analytics_agent)
    except Exception as e:
        logger.error(f"Error connecting to the portfolio analytics agent: {e}")
        return "Error connecting to the portfolio analytics agent."


@tool
async def performance_tracking_tool(message: str) -> str | None:
    """Sends a message to the performance tracking agent and returns the response."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SETTINGS) as httpx_client:
            global performance_tracking_agent
            performance_tracking_agent = await A2AClient.get_client_from_agent_card_url(
                httpx_client,
                "https://performance-tracking-agent-906909573150.us-central1.run.app",
                http_kwargs={
                    "auth": (
                        TRACKING_AUTH_USERNAME,
                        TRACKING_AUTH_PASSWORD,
                    )
                },
            )
            logger.info("✅ Connected to the performance tracking agent.")
            return await _send_message_to_a2a_agent(message, performance_tracking_agent)
    except Exception as e:
        logger.error(f"Error connecting to the performance tracking agent: {e}")
        return "Error connecting to the performance tracking agent."


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

    """

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more information to complete the request."
        "Set response status to error if there is an error while processing the request."
        "Set response status to completed if the request is complete."
        "Record what you are thinking and next steps in the thought field."
    )

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.tools = [
            customer_portfolio_tool,
            risk_calculation_tool,
            portfolio_analytics_tool,
            performance_tracking_tool,
        ]
        logger.info("🏃‍➡️ Creating Portfolio Copilot Agent...")
        logger.info(f"Using tools: {_get_a2a_descriptions()}")
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            name="portfolio_copilot_agent",
            prompt="You are a helpful portfolio copilot agent.",
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
            checkpointer=InMemorySaver(),
            # store=InMemoryStore(),
        )

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
