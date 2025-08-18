import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Import your agent (adjust the import path as needed)
from agent import PortfolioCopilotAgent

# Page configuration
st.set_page_config(
    page_title="Portfolio Copilot",
    page_icon="💼",
    layout="centered",
)

# Simple CSS for chat messages
st.markdown(
    """
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .agent-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit_{uuid.uuid4().hex[:8]}"


class SyncAgentWrapper:
    """Synchronous wrapper for the async agent"""

    def __init__(self, agent):
        self.agent = agent
        self._loop = None
        self._setup_loop()

    def _setup_loop(self):
        """Set up a dedicated event loop for this wrapper"""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def invoke(self, messages: dict) -> Dict[str, Any]:
        """Synchronous invoke method"""
        try:
            if self._loop is None or self._loop.is_closed():
                self._setup_loop()

            if self._loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()
                return asyncio.run(self._async_invoke(messages))
            else:
                return self._loop.run_until_complete(self._async_invoke(messages))

        except ImportError:
            try:
                return asyncio.run(self._async_invoke(messages))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(self._async_invoke(messages))
                finally:
                    loop.close()
        except Exception as e:
            return {
                "messages": [
                    {
                        "content": json.dumps(
                            {
                                "status": "error",
                                "thought": f"Sync wrapper error: {str(e)}",
                                "message": f"An error occurred: {str(e)}",
                            }
                        )
                    }
                ]
            }

    async def _async_invoke(self, messages: dict):
        """Internal async invoke method"""
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        return await self.agent.ainvoke_with_tracing(messages, config=config)


def initialize_agent():
    """Initialize the Portfolio Copilot Agent"""
    try:
        if not st.session_state.agent_initialized:
            with st.spinner("Initializing agent..."):
                base_agent = PortfolioCopilotAgent()
                st.session_state.agent = SyncAgentWrapper(base_agent)
                st.session_state.agent_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return False


def get_agent_response(user_input: str) -> Dict[str, Any]:
    """Get response from the agent synchronously"""
    try:
        if st.session_state.agent is None:
            return {
                "status": "error",
                "thought": "Agent not initialized",
                "message": "The agent is not properly initialized. Please refresh the page and try again.",
            }

        response = st.session_state.agent.invoke({"messages": [("user", user_input)]})

        if "messages" in response and len(response["messages"]) > 0:
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                try:
                    parsed_response = json.loads(last_message.content)
                    if isinstance(parsed_response, dict):
                        return parsed_response
                except json.JSONDecodeError:
                    pass

                return {
                    "status": "completed",
                    "thought": "Processing user request",
                    "message": last_message.content,
                }

        return {
            "status": "error",
            "thought": "No valid response received",
            "message": "Sorry, I couldn't process your request properly.",
        }
    except Exception as e:
        return {
            "status": "error",
            "thought": f"Error occurred: {str(e)}",
            "message": f"An error occurred while processing your request: {str(e)}",
        }


def main():
    # Title
    st.title("💼 Portfolio Copilot")

    # Initialize agent if not already done
    if not st.session_state.agent_initialized:
        if not initialize_agent():
            st.stop()

    # Chat container
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message.get("is_user", False):
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message.get("message", "No response"))

    # Chat input
    if prompt := st.chat_input("Ask about your portfolio..."):
        # Add user message to chat history
        user_message = {
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_user": True,
        }
        st.session_state.chat_history.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent_response = get_agent_response(prompt)

            # Display response
            st.write(agent_response.get("message", "No response"))

            # Add to history
            agent_response["timestamp"] = datetime.now().strftime("%H:%M:%S")
            agent_response["is_user"] = False
            st.session_state.chat_history.append(agent_response)


if __name__ == "__main__":
    main()
