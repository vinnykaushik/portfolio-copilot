import os
import traceback
from typing import Any
from uuid import uuid4

from a2a.client import A2AClient
from a2a.types import (
    SendMessageResponse,
    GetTaskResponse,
    SendMessageSuccessResponse,
    Task,
    TaskState,
    SendMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
    SendStreamingMessageRequest,
)
import httpx

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8888")

# Increase timeout settings
TIMEOUT_SETTINGS = httpx.Timeout(
    connect=30.0,  # 30 seconds to connect
    read=300.0,  # 5 minutes to read response
    write=30.0,  # 30 seconds to write
    pool=30.0,  # 30 seconds for pool operations
)


def create_send_message_payload(
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


def print_json_response(response: Any, description: str) -> None:
    """Helper function to print the JSON representation of a response."""
    print(f"--- {description} ---")
    if hasattr(response, "root"):
        print(f"{response.root.model_dump_json(exclude_none=True)}\n")
    else:
        print(f"{response.model_dump(mode='json', exclude_none=True)}\n")


async def run_single_turn_test(client: A2AClient) -> None:
    """Runs a single-turn non-streaming test."""
    print("--- ✉️  Single Turn Request ---")
    print("⏳ This may take a few minutes for complex portfolio calculations...")

    send_message_payload = create_send_message_payload(
        text="""analyze the following portfolio
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
                                        {
                                            "type": "bonds",
                                            "name": "Corporate Bond Index Fund",
                                            "value": 45000,
                                        },
                                    ],
                                    "last_updated": "2025-06-18",
                                }"""
    )
    request = SendMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_message_payload)
    )

    try:
        # Send Message with timeout handling
        response: SendMessageResponse = await client.send_message(request)
        print_json_response(response, "📥 Single Turn Request Response")

        if not isinstance(response.root, SendMessageSuccessResponse):
            print("received non-success response. Aborting get task ")
            return

        if not isinstance(response.root.result, Task):
            print("received non-task response. Aborting get task ")
            return

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
                print_json_response(get_response, "📥 Query Task Response")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Attempt {attempt + 1} failed, retrying... ({e})")
                    await asyncio.sleep(2)  # Wait 2 seconds before retry
                else:
                    raise

    except Exception as e:
        print(f"❌ Error in single turn test: {e}")
        if "timeout" in str(e).lower():
            print(
                "💡 This appears to be a timeout. The calculation may still be running."
            )
            print("💡 Consider using streaming mode or increasing timeout values.")


async def run_streaming_test(client: A2AClient) -> None:
    """Runs a single-turn streaming test."""
    print("--- ⏩ Single Turn Streaming Request ---")
    print("⏳ Streaming responses (this may take a while)...")

    send_message_payload = create_send_message_payload(
        text="Analyze a portfolio with 100 AAPL and 100 GOOGL. This is all that's in the portfolio.",
    )

    request = SendStreamingMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_message_payload)
    )

    try:
        stream_response = client.send_message_streaming(request)
        chunk_count = 0
        async for chunk in stream_response:
            chunk_count += 1
            print(f"--- ⏳ Streaming Chunk {chunk_count} ---")
            print_json_response(chunk, f"Chunk {chunk_count}")

        print(f"✅ Streaming completed with {chunk_count} chunks")

    except Exception as e:
        print(f"❌ Error in streaming test: {e}")
        if "timeout" in str(e).lower():
            print("💡 Consider increasing timeout or checking server performance.")


async def run_multi_turn_test(client: A2AClient) -> None:
    """Runs a multi-turn non-streaming test."""
    print("--- 📝 Multi-Turn Request ---")
    print("⏳ This may take a few minutes...")

    # --- First Turn ---
    first_turn_payload = create_send_message_payload(
        text="Can you analyze the following portfolio?"
    )
    request1 = SendMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**first_turn_payload)
    )

    try:
        first_turn_response: SendMessageResponse = await client.send_message(request1)
        print_json_response(first_turn_response, "📥 Multi-Turn: First Turn Response")

        context_id: str | None = None
        if isinstance(
            first_turn_response.root, SendMessageSuccessResponse
        ) and isinstance(first_turn_response.root.result, Task):
            task: Task = first_turn_response.root.result
            context_id = task.contextId  # Capture context ID

            # --- Second Turn (if input required) ---
            if task.status.state == TaskState.input_required and context_id:
                print("--- 📝 Multi-Turn: Second Turn (Input Required) ---")
                second_turn_payload = create_send_message_payload(
                    "100 AAPL 300 GOOGL. That's all that's in the portfolio",
                    task.id,
                    context_id,
                )
                request2 = SendMessageRequest(
                    id=str(uuid4()), params=MessageSendParams(**second_turn_payload)
                )
                second_turn_response = await client.send_message(request2)
                print_json_response(
                    second_turn_response, "Multi-Turn: Second Turn Response"
                )
            elif not context_id:
                print(
                    "--- ⚠️ Warning: Could not get context ID from first turn response. ---"
                )
            else:
                print(
                    "--- 🚀 First turn completed, no further input required for this test case. ---"
                )

    except Exception as e:
        print(f"❌ Error in multi-turn test: {e}")
        if "timeout" in str(e).lower():
            print("💡 The calculation may still be running. Check server logs.")


async def main() -> None:
    """Main function to run the tests."""
    print(f"--- 🔄 Connecting to agent at {AGENT_URL}... ---")
    try:
        # Create HTTP client with extended timeout
        async with httpx.AsyncClient(timeout=TIMEOUT_SETTINGS) as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, AGENT_URL
            )
            print("--- ✅ Connection successful. ---")
            print(f"⚙️ Using extended timeouts: {TIMEOUT_SETTINGS}")

            # Run tests with proper exception handling
            try:
                await run_single_turn_test(client)
            except Exception as e:
                print(f"❌ Single turn test failed: {e}")

            try:
                await run_streaming_test(client)
            except Exception as e:
                print(f"❌ Streaming test failed: {e}")

            try:
                await run_multi_turn_test(client)
            except Exception as e:
                print(f"❌ Multi-turn test failed: {e}")

    except Exception as e:
        traceback.print_exc()
        print(f"--- ❌ An error occurred: {e} ---")
        print("Ensure the agent server is running.")

        # Additional debugging info
        if "timeout" in str(e).lower():
            print("\n💡 TIMEOUT TROUBLESHOOTING:")
            print("1. Check if your MCP server is running properly")
            print("2. Monitor server logs for processing time")
            print("3. Consider using streaming mode for long-running calculations")
            print("4. Verify your risk calculation tools are responding quickly")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
