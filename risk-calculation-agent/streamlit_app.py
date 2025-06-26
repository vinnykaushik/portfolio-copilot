import streamlit as st
import asyncio
import json
from dotenv import load_dotenv
from agent import agent, tool_descriptions, get_agent_response, extract_response_content

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Risk Calculation Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better styling
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
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: auto;
        border-left: 4px solid #4caf50;
    }
    
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    
    .example-portfolio {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent_and_tools():
    """Get the pre-initialized agent and tool descriptions."""
    return agent, tool_descriptions


def display_message(message: str, is_user: bool = True):
    """Display a chat message with custom styling."""
    if is_user:
        st.markdown(
            f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>
            {message}
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="chat-message assistant-message">
            <strong>Risk Agent:</strong><br>
            {message}
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>📊 Risk Calculation Agent</h1>
        <p>Analyze financial portfolios and calculate risk metrics</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Get agent and tools
    with st.spinner("Loading Risk Calculation Agent..."):
        risk_agent, agent_tool_descriptions = get_agent_and_tools()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "risk_agent" not in st.session_state:
        st.session_state.risk_agent = risk_agent

    # Sidebar with information
    with st.sidebar:
        st.header("📋 Available Tools")
        with st.expander("View Tool Descriptions"):
            st.markdown(agent_tool_descriptions)

        st.header("💡 Example Portfolio")
        st.markdown(
            """
        <div class="example-portfolio">
        <strong>Sample Portfolio JSON:</strong><br>
        <code>
        {<br>
        &nbsp;&nbsp;"customer_id": "33333",<br>
        &nbsp;&nbsp;"customer_name": "Carl Pei",<br>
        &nbsp;&nbsp;"portfolio_value": 320000,<br>
        &nbsp;&nbsp;"investments": [<br>
        &nbsp;&nbsp;&nbsp;&nbsp;{<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"type": "stocks",<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"symbol": "NVDA",<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"quantity": 40<br>
        &nbsp;&nbsp;&nbsp;&nbsp;},<br>
        &nbsp;&nbsp;&nbsp;&nbsp;{<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"type": "mutual_funds",<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"name": "Technology Select Sector SPDR Fund",<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"value": 90000<br>
        &nbsp;&nbsp;&nbsp;&nbsp;}<br>
        &nbsp;&nbsp;],<br>
        &nbsp;&nbsp;"last_updated": "2025-06-18"<br>
        }
        </code>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message(message["content"], message["role"] == "user")

        # Chat input
        if prompt := st.chat_input("Ask about portfolio risk calculations..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            display_message(prompt, is_user=True)

            # Get agent response
            with st.spinner("Calculating..."):
                response = asyncio.run(get_agent_response(prompt))

                # Extract the actual response content
                response_content = extract_response_content(response)

            # Add assistant response to history
            st.session_state.messages.append(
                {"role": "assistant", "content": response_content}
            )

            # Display assistant response
            display_message(response_content, is_user=False)

            # Rerun to update the display
            st.rerun()

    with col2:
        st.header("🚀 Quick Actions")

        if st.button("📈 Calculate Expected Return", use_container_width=True):
            sample_query = (
                "What is the expected return of this portfolio? "
                + json.dumps(
                    {
                        "customer_id": "33333",
                        "customer_name": "Carl Pei",
                        "portfolio_value": 320000,
                        "investments": [
                            {"type": "stocks", "symbol": "NVDA", "quantity": 40},
                            {"type": "stocks", "symbol": "AMD", "quantity": 18},
                            {
                                "type": "mutual_funds",
                                "name": "Technology Select Sector SPDR Fund",
                                "value": 90000,
                            },
                            {"type": "stocks", "symbol": "INTC", "quantity": 25},
                        ],
                        "last_updated": "2025-06-18",
                    }
                )
            )
            st.session_state.messages.append({"role": "user", "content": sample_query})
            st.rerun()

        if st.button("⚠️ Calculate Risk Metrics", use_container_width=True):
            sample_query = "Calculate the risk metrics for this portfolio including VaR and volatility"
            st.session_state.messages.append({"role": "user", "content": sample_query})
            st.rerun()

        if st.button("📊 Portfolio Analysis", use_container_width=True):
            sample_query = "Provide a comprehensive analysis of the portfolio including diversification and recommendations"
            st.session_state.messages.append({"role": "user", "content": sample_query})
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666;">
        <small>Risk Calculation Agent powered by LangGraph and Google Gemini</small>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
