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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling with improved contrast
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
        color: #333333;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #f8f9ff;
        border-left-color: #2196f3;
        color: #1a1a1a;
    }
    .agent-message {
        background-color: #fafafa;
        border-left-color: #9c27b0;
        color: #1a1a1a;
    }
    .status-completed {
        color: #2e7d32;
        font-weight: bold;
    }
    .status-input-required {
        color: #ef6c00;
        font-weight: bold;
    }
    .status-error {
        color: #c62828;
        font-weight: bold;
    }
    .agent-thinking {
        background-color: #fff8e1;
        padding: 0.5rem;
        border-radius: 5px;
        font-style: italic;
        margin: 0.5rem 0;
        color: #333333;
        border: 1px solid #ffcc02;
    }
    /* Override Streamlit's default text colors */
    .chat-message p {
        color: inherit !important;
        margin: 0;
    }
    .chat-message strong {
        color: inherit !important;
    }
    /* Ensure proper text rendering */
    .stMarkdown {
        color: inherit;
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
    # Create a unique thread ID for this Streamlit session
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
            # Try to get existing loop
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new loop if none exists
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def invoke(self, messages: dict) -> Dict[str, Any]:
        """Synchronous invoke method"""
        try:
            # Ensure we have a valid loop
            if self._loop is None or self._loop.is_closed():
                self._setup_loop()

            # Run the async method synchronously
            if self._loop.is_running():
                # If loop is already running, we need to handle this differently
                import nest_asyncio

                nest_asyncio.apply()
                return asyncio.run(self._async_invoke(messages))
            else:
                return self._loop.run_until_complete(self._async_invoke(messages))

        except ImportError:
            # Fallback if nest_asyncio is not available
            try:
                return asyncio.run(self._async_invoke(messages))
            except RuntimeError:
                # Last resort: create completely new loop
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
        # Use the session-specific thread ID for checkpointer
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        return await self.agent.ainvoke_with_tracing(messages, config=config)


def initialize_agent():
    """Initialize the Portfolio Copilot Agent"""
    try:
        if not st.session_state.agent_initialized:
            with st.spinner("Initializing Portfolio Copilot Agent..."):
                base_agent = PortfolioCopilotAgent()
                # Wrap the agent with our sync wrapper
                st.session_state.agent = SyncAgentWrapper(base_agent)
                st.session_state.agent_initialized = True
            st.success("Agent initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return False


def get_agent_response(user_input: str) -> Dict[str, Any]:
    """Get response from the agent synchronously"""
    try:
        # Check if agent is properly initialized
        if st.session_state.agent is None:
            return {
                "status": "error",
                "thought": "Agent not initialized",
                "message": "The agent is not properly initialized. Please refresh the page and try again.",
            }

        # Invoke the agent synchronously
        response = st.session_state.agent.invoke({"messages": [("user", user_input)]})

        # Extract the structured response
        if "messages" in response and len(response["messages"]) > 0:
            last_message = response["messages"][-1]
            if hasattr(last_message, "content"):
                # Try to parse as JSON if it's a structured response
                try:
                    parsed_response = json.loads(last_message.content)
                    if isinstance(parsed_response, dict):
                        return parsed_response
                except json.JSONDecodeError:
                    pass

                # If not JSON, create a basic response structure
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


def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling"""
    if is_user:
        # Escape HTML in user content to prevent XSS
        content = message["content"].replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {content}
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        status_class = f"status-{message.get('status', 'completed').replace('_', '-')}"
        status_emoji = {"completed": "✅", "input_required": "⏳", "error": "❌"}.get(
            message.get("status", "completed"), "🤖"
        )

        # Escape HTML in agent content
        agent_message = (
            message.get("message", "No response available")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        agent_thought = (
            message.get("thought", "").replace("<", "&lt;").replace(">", "&gt;")
        )

        st.markdown(
            f"""
        <div class="chat-message agent-message">
            <strong>{status_emoji} Portfolio Copilot:</strong>
            <span class="{status_class}">({message.get('status', 'completed').replace('_', ' ').title()})</span><br>
            {agent_message}
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Show agent's thinking process if available
        if agent_thought:
            st.markdown(
                f"""
            <div class="agent-thinking">
                <strong>💭 Agent's thinking:</strong> {agent_thought}
            </div>
            """,
                unsafe_allow_html=True,
            )


def main():
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>💼 Portfolio Copilot</h1>
        <p>Your AI-powered portfolio management assistant</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar with information
    with st.sidebar:
        st.header("📊 Agent Information")

        if st.session_state.agent_initialized:
            st.success("✅ Agent Ready")
        else:
            st.warning("⚠️ Agent Not Initialized")

        st.markdown(
            """
        ### Available Tools:
        - **Customer Portfolio Tool** (Port 7777)
        - **Risk Calculation Tool** (Port 9999)  
        - **Portfolio Analytics Tool** (Port 8888)
        
        ### How to Use:
        1. Ask questions about your portfolio
        2. Request risk calculations
        3. Get portfolio analytics
        4. The agent will use appropriate tools automatically
        """
        )

        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("🔄 Reinitialize Agent"):
            st.session_state.agent = None
            st.session_state.agent_initialized = False
            st.rerun()

    # Initialize agent if not already done
    if not st.session_state.agent_initialized:
        if not initialize_agent():
            st.stop()

    # Main chat interface
    st.header("💬 Chat Interface")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message, message.get("is_user", False))

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_area(
                "Enter your message:",
                placeholder="Ask me about your portfolio, risk calculations, or analytics...",
                height=100,
                key="user_input",
            )

        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)

    # Process user input
    if submit_button and user_input.strip():
        # Check if agent is initialized before processing
        if not st.session_state.agent_initialized or st.session_state.agent is None:
            st.error("Agent is not initialized. Please refresh the page.")
            st.stop()

        # Add user message to chat history
        user_message = {
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_user": True,
        }
        st.session_state.chat_history.append(user_message)

        # Show processing indicator
        with st.spinner("🤔 Agent is thinking..."):
            # Get agent response synchronously - no threading needed!
            agent_response = get_agent_response(user_input)

        # Add agent response to chat history
        agent_response["timestamp"] = datetime.now().strftime("%H:%M:%S")
        agent_response["is_user"] = False
        st.session_state.chat_history.append(agent_response)

        # Rerun to update the display
        st.rerun()

    # Sample queries
    if not st.session_state.chat_history:
        st.header("💡 Sample Queries")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Get Portfolio Overview", use_container_width=True):
                st.session_state.chat_history.append(
                    {
                        "content": "Can you provide an overview of my current portfolio?",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "is_user": True,
                    }
                )
                st.rerun()

        with col2:
            if st.button("⚠️ Calculate Risk Metrics", use_container_width=True):
                st.session_state.chat_history.append(
                    {
                        "content": "What are the current risk metrics for my portfolio?",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "is_user": True,
                    }
                )
                st.rerun()

        with col3:
            if st.button("📈 Portfolio Analytics", use_container_width=True):
                st.session_state.chat_history.append(
                    {
                        "content": "Can you analyze my portfolio performance?",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "is_user": True,
                    }
                )
                st.rerun()


if __name__ == "__main__":
    main()
