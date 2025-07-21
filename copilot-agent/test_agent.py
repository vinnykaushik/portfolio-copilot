import streamlit as st
from typing import Dict, Literal, Any, List, Annotated, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import json
import uuid
import asyncio
import os
from datetime import datetime
import difflib
from html import escape
from dotenv import load_dotenv
import re

# Import your existing portfolio agent
from agent import PortfolioCopilotAgent

# Load environment variables
load_dotenv()

# ===== STATE DEFINITIONS =====


class PortfolioAgentState(TypedDict):
    """Base state from the portfolio agent"""

    messages: Annotated[List[BaseMessage], add_messages]
    status: Literal["input_required", "completed", "error"]
    thought: str
    message: str


class EvaluationState(PortfolioAgentState):
    """Extended state for evaluation"""

    # Test configuration
    user_tone: str
    test_scenario: str
    client_name: str

    # Conversation tracking
    chatbot_history: str
    advisor_query: str
    test_complete: bool

    # Evaluation metrics
    evaluation_score: int
    evaluation_criteria: str
    evaluation_evidence: str

    # Optimization
    original_prompt: str
    optimized_prompt: str
    optimization_needed: bool
    prompt_diff: Dict[str, Any]

    # Test validation
    expected_tools_used: List[str]
    actual_tools_used: List[str]
    conversation_turns: int
    max_turns: int


# ===== TEST SCENARIOS =====

ADVISOR_TEST_SCENARIOS = {
    "client_portfolio_concentration": {
        "initial_message": "get me carl pei's portfolio and run portfolio concentration analysis on it",
        "client_name": "carl pei",
        "expected_tools": ["customer_portfolio_tool", "portfolio_analytics_tool"],
        "validation_points": [
            "retrieves_correct_client_portfolio",
            "identifies_concentration_risks",
            "provides_diversification_recommendations",
        ],
        "follow_up_questions": [
            "What sectors are overweighted?",
            "Should we rebalance before year end?",
            "What are the tax implications?",
        ],
    },
    "risk_assessment_for_client": {
        "initial_message": "What's the current risk profile for sarah chen? She mentioned wanting to retire in 5 years",
        "client_name": "sarah chen",
        "expected_tools": ["customer_portfolio_tool", "risk_calculation_tool"],
        "validation_points": [
            "retrieves_client_portfolio",
            "calculates_risk_metrics",
            "considers_retirement_timeline",
            "provides_risk_adjusted_recommendations",
        ],
        "follow_up_questions": [
            "Is her current allocation appropriate for her timeline?",
            "What changes would you recommend?",
            "Show me a more conservative allocation",
        ],
    },
    "performance_comparison": {
        "initial_message": "Compare james wilson's YTD performance against the S&P 500 and his peer group",
        "client_name": "james wilson",
        "expected_tools": [
            "customer_portfolio_tool",
            "performance_tracking_tool",
            "portfolio_analytics_tool",
        ],
        "validation_points": [
            "retrieves_client_performance_data",
            "fetches_benchmark_data",
            "calculates_relative_performance",
            "identifies_outperformance_underperformance",
        ],
        "follow_up_questions": [
            "What's driving the underperformance?",
            "Which holdings contributed most to returns?",
            "How does this compare to last year?",
        ],
    },
    "urgent_risk_alert": {
        "initial_message": "One of my clients, michael torres, just called worried about his NVDA position. What's his exposure and should we trim?",
        "client_name": "michael torres",
        "expected_tools": [
            "customer_portfolio_tool",
            "risk_calculation_tool",
            "portfolio_analytics_tool",
        ],
        "validation_points": [
            "quickly_retrieves_position_data",
            "calculates_portfolio_impact",
            "provides_actionable_recommendation",
            "considers_tax_implications",
        ],
        "follow_up_questions": [
            "What would trimming 50% do to his overall allocation?",
            "Are there other tech positions we should review?",
            "Draft an email summary for the client",
        ],
    },
}

ADVISOR_TONES = [
    "Experienced Senior Advisor",
    "New Junior Advisor",
    "Urgent Client Meeting",
    "Compliance-Focused",
    "Tech-Savvy Advisor",
    "Relationship Manager",
    "Stressed During Market Volatility",
    "Detail-Oriented Analyst",
]

# ===== LLM INITIALIZATION =====


def get_llm():
    """Initialize LLM based on available API keys"""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4", temperature=0.7)
    elif os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
    else:
        raise ValueError("No LLM API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY")


# ===== NODE IMPLEMENTATIONS =====


async def start_conversation_node(state: EvaluationState) -> Dict:
    """Initialize the conversation with the test scenario"""
    scenario = ADVISOR_TEST_SCENARIOS[state["test_scenario"]]
    initial_message = scenario["initial_message"]

    return {
        "messages": [HumanMessage(content=initial_message)],
        "chatbot_history": f"Advisor: {initial_message}",
        "advisor_query": initial_message,
        "client_name": scenario["client_name"],
        "expected_tools_used": scenario["expected_tools"],
        "conversation_turns": 0,
        "max_turns": 10,
        "test_complete": False,
    }


async def portfolio_agent_node(state: EvaluationState) -> Dict:
    """Run the portfolio copilot agent"""
    # Get the last message
    last_message = state["messages"][-1].content if state["messages"] else ""

    # Initialize and run the portfolio agent
    agent = PortfolioCopilotAgent()

    try:
        # Run the agent
        result = await agent.ainvoke_with_tracing(
            {"messages": [("user", last_message)]},
            {"configurable": {"thread_id": state.get("thread_id", str(uuid.uuid4()))}},
        )

        # Extract response
        if "messages" in result and len(result["messages"]) > 0:
            agent_message = result["messages"][-1]
            if hasattr(agent_message, "content"):
                try:
                    parsed_response = json.loads(agent_message.content)
                    response_text = parsed_response.get(
                        "message", agent_message.content
                    )
                    status = parsed_response.get("status", "completed")
                    thought = parsed_response.get("thought", "")
                except json.JSONDecodeError:
                    response_text = agent_message.content
                    status = "completed"
                    thought = ""
            else:
                response_text = str(agent_message)
                status = "completed"
                thought = ""
        else:
            response_text = "No response from agent"
            status = "error"
            thought = "Failed to get response"

        # Track which tools were used (you might need to extract this from the agent's execution)
        tools_used = extract_tools_used(thought)

        return {
            "messages": [AIMessage(content=response_text)],
            "chatbot_history": state["chatbot_history"] + f"\nAgent: {response_text}",
            "status": status,
            "thought": thought,
            "message": response_text,
            "actual_tools_used": state.get("actual_tools_used", []) + tools_used,
            "conversation_turns": state["conversation_turns"] + 1,
        }

    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")],
            "status": "error",
            "thought": f"Error occurred: {str(e)}",
            "message": f"I encountered an error: {str(e)}",
            "conversation_turns": state["conversation_turns"] + 1,
        }


def simulate_advisor_node(state: EvaluationState) -> Dict:
    """Simulate financial advisor responses based on tone and scenario"""
    llm = get_llm()

    # Get scenario details
    scenario = ADVISOR_TEST_SCENARIOS[state["test_scenario"]]
    follow_ups = scenario.get("follow_up_questions", [])

    # Check if we should end the conversation
    if (
        state["conversation_turns"] >= state["max_turns"]
        or state["status"] == "completed"
    ):
        if state["conversation_turns"] < 3:  # Ensure minimum conversation
            # Ask a follow-up question
            question_idx = state["conversation_turns"] - 1
            if question_idx < len(follow_ups):
                response = follow_ups[question_idx]
            else:
                response = "Thanks, that's helpful. Can you summarize the key points?"
        else:
            response = "Thank you, this is exactly what I needed for my client meeting."
            state["test_complete"] = True
    else:
        # Generate contextual response based on advisor tone
        prompt = f"""You are a {state['user_tone']} asking about {state['client_name']}'s portfolio.
        
The agent just said: {state['messages'][-1].content if state['messages'] else 'Nothing yet'}

Scenario context: {state['test_scenario']}

Generate a realistic follow-up response or question that this type of advisor would ask.
Keep it concise and focused on getting actionable insights for the client.

Response:"""

        response = llm.invoke(prompt).content

    return {
        "messages": [HumanMessage(content=response)],
        "chatbot_history": state["chatbot_history"] + f"\nAdvisor: {response}",
        "test_complete": state.get("test_complete", False),
    }


def evaluation_node(state: EvaluationState) -> Dict:
    """Evaluate the conversation quality"""
    llm = get_llm()

    evaluation_prompt = f"""Evaluate this financial advisor portfolio analysis conversation.

CONVERSATION:
{state['chatbot_history']}

EXPECTED TOOLS: {state['expected_tools_used']}
ACTUAL TOOLS USED: {state.get('actual_tools_used', [])}
CLIENT NAME: {state['client_name']}
SCENARIO: {state['test_scenario']}

Score the agent on these criteria (0-10):

1. **Client Data Accuracy**: Did the agent retrieve the correct client's portfolio?
2. **Analysis Completeness**: Were all requested analyses performed?
3. **Tool Usage**: Were the appropriate tools used for the analysis?
4. **Actionability**: Were recommendations specific and actionable?
5. **Professional Communication**: Was the language appropriate for financial advisors?
6. **Response Relevance**: Did the agent stay on topic and address the advisor's needs?
7. **Efficiency**: Was the information provided concisely without unnecessary details?

Scoring guidelines:
- 9-10: Exceptional performance, all criteria met perfectly
- 7-8: Good performance, minor improvements possible  
- 5-6: Adequate but needs improvement in multiple areas
- 3-4: Poor performance, major issues present
- 0-2: Failed to meet basic requirements

Respond in this format:
Score: [0-10]
Criteria: [List criteria that scored poorly]
Evidence: [Specific examples from conversation]
"""

    result = llm.invoke(evaluation_prompt).content

    # Parse the evaluation result
    score_match = re.search(r"Score:\s*(\d+)", result)  # type: ignore
    criteria_match = re.search(r"Criteria:\s*(.+?)(?=Evidence:|$)", result, re.DOTALL)  # type: ignore
    evidence_match = re.search(r"Evidence:\s*(.+)", result, re.DOTALL)  # type: ignore

    score = int(score_match.group(1)) if score_match else 5
    criteria = (
        criteria_match.group(1).strip() if criteria_match else "No criteria provided"
    )
    evidence = (
        evidence_match.group(1).strip() if evidence_match else "No evidence provided"
    )

    return {
        "evaluation_score": score,
        "evaluation_criteria": criteria,
        "evaluation_evidence": evidence,
        "optimization_needed": score < 8,
    }


async def optimization_node(state: EvaluationState) -> Dict:
    """Optimize prompts based on evaluation results"""
    if not state.get("optimization_needed", False):
        return {"optimized_prompt": state.get("original_prompt", ""), "prompt_diff": {}}

    llm = get_llm()

    # Get the original prompt (you'll need to implement this based on your agent)
    original_prompt = get_agent_prompt()

    optimization_prompt = f"""Analyze this portfolio agent conversation and suggest prompt improvements.

CONVERSATION:
{state['chatbot_history']}

EVALUATION SCORE: {state['evaluation_score']}/10
ISSUES IDENTIFIED: {state['evaluation_criteria']}
EVIDENCE: {state['evaluation_evidence']}

CURRENT AGENT PROMPT:
{original_prompt}

Suggest specific improvements to the prompt that would address the identified issues.
Focus on:
1. Improving client data retrieval accuracy
2. Ensuring complete analysis coverage
3. Making responses more actionable for advisors
4. Optimizing for the financial advisory use case

Provide the optimized prompt maintaining the same structure and format as the original.

OPTIMIZED PROMPT:"""

    optimized = llm.invoke(optimization_prompt).content

    # Generate diff
    diff = generate_prompt_diff(original_prompt, optimized)  # type: ignore

    return {
        "original_prompt": original_prompt,
        "optimized_prompt": optimized,
        "prompt_diff": diff,
    }


def generate_report_node(state: EvaluationState) -> Dict:
    """Generate final evaluation report"""
    report = f"""
# Portfolio Agent Evaluation Report

## Test Configuration
- **Scenario**: {state['test_scenario']}
- **Advisor Tone**: {state['user_tone']}
- **Client**: {state['client_name']}
- **Conversation Turns**: {state['conversation_turns']}

## Evaluation Results
- **Score**: {state['evaluation_score']}/10
- **Issues**: {state['evaluation_criteria']}
- **Evidence**: {state['evaluation_evidence']}

## Tool Usage
- **Expected**: {', '.join(state['expected_tools_used'])}
- **Actual**: {', '.join(state.get('actual_tools_used', []))}

## Optimization
- **Needed**: {'Yes' if state.get('optimization_needed') else 'No'}
"""

    return {"message": report}


# ===== HELPER FUNCTIONS =====


def extract_tools_used(thought: str) -> List[str]:
    """Extract which tools were used from the agent's thought process"""
    tools = []
    tool_patterns = [
        "customer_portfolio_tool",
        "risk_calculation_tool",
        "portfolio_analytics_tool",
        "performance_tracking_tool",
    ]

    for tool in tool_patterns:
        if tool in thought.lower():
            tools.append(tool)

    return tools


def get_agent_prompt() -> str:
    """Get the current agent prompt (implement based on your agent structure)"""
    # This is a placeholder - you'll need to extract the actual prompt from your agent
    return """You are a helpful portfolio copilot agent for financial advisors.
Help advisors analyze their clients' portfolios and provide actionable insights."""


def generate_prompt_diff(original: str, optimized: str) -> Dict[str, Any]:
    """Generate a diff between original and optimized prompts"""
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            optimized.splitlines(keepends=True),
            fromfile="original_prompt",
            tofile="optimized_prompt",
        )
    )

    return {"diff_lines": diff_lines, "has_changes": len(diff_lines) > 0}


def get_diff_html(original: str, optimized: str) -> str:
    """Generate HTML diff for Streamlit display"""
    differ = difflib.Differ()
    diff = list(differ.compare(original.splitlines(), optimized.splitlines()))

    result = ""
    for line in diff:
        content = escape(line[2:].strip())

        if line.startswith("? "):
            continue
        elif line.startswith("- "):
            result += f"<div style='color:red;'>➖ {content}</div>"
        elif line.startswith("+ "):
            result += f"<div style='color:green;'>➕ {content}</div>"
        else:
            result += f"<div>{content}</div>"

    return result


# ===== GRAPH CONSTRUCTION =====


def should_continue_conversation(
    state: EvaluationState,
) -> Literal["continue", "evaluate"]:
    """Determine whether to continue the conversation or move to evaluation"""
    if (
        state.get("test_complete", False)
        or state["conversation_turns"] >= state["max_turns"]
    ):
        return "evaluate"
    return "continue"


def build_evaluation_graph():
    """Build the evaluation graph"""
    builder = StateGraph(EvaluationState)

    # Add nodes
    builder.add_node("start_conversation", start_conversation_node)
    builder.add_node("portfolio_agent", portfolio_agent_node)
    builder.add_node("simulate_advisor", simulate_advisor_node)
    builder.add_node("evaluate", evaluation_node)
    builder.add_node("optimize", optimization_node)
    builder.add_node("generate_report", generate_report_node)

    # Define flow
    builder.add_edge(START, "start_conversation")
    builder.add_edge("start_conversation", "portfolio_agent")
    builder.add_edge("portfolio_agent", "simulate_advisor")

    # Conditional routing
    builder.add_conditional_edges(
        "simulate_advisor",
        should_continue_conversation,
        {"continue": "portfolio_agent", "evaluate": "evaluate"},
    )

    builder.add_edge("evaluate", "optimize")
    builder.add_edge("optimize", "generate_report")
    builder.add_edge("generate_report", END)

    # Compile with checkpointer
    return builder.compile(checkpointer=MemorySaver())


# ===== STREAMLIT UI =====


def main():
    st.set_page_config(
        page_title="Portfolio Agent Evaluation System", page_icon="📊", layout="wide"
    )

    st.title("📊 Financial Advisor Portfolio Agent Evaluation")
    st.markdown("Test and optimize the portfolio analysis agent for financial advisors")

    # Initialize session state
    if "eval_graph" not in st.session_state:
        st.session_state.eval_graph = build_evaluation_graph()
    if "test_results" not in st.session_state:
        st.session_state.test_results = []

    # Sidebar configuration
    with st.sidebar:
        st.header("Test Configuration")

        # Scenario selection
        scenario = st.selectbox(
            "Select Test Scenario",
            options=list(ADVISOR_TEST_SCENARIOS.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )

        # Tone selection
        tone = st.selectbox("Select Advisor Persona", options=ADVISOR_TONES)

        st.markdown("---")

        # Test scenario details
        st.subheader("Scenario Details")
        scenario_details = ADVISOR_TEST_SCENARIOS[scenario]
        st.write(f"**Client**: {scenario_details['client_name']}")
        st.write(f"**Expected Tools**: {', '.join(scenario_details['expected_tools'])}")

        st.markdown("---")

        if st.button("🔄 Clear Results"):
            st.session_state.test_results = []
            st.rerun()

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Evaluation Console")

        if st.button("▶️ Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                # Prepare initial state
                thread_id = str(uuid.uuid4())
                initial_state = {
                    "messages": [],
                    "user_tone": tone,
                    "test_scenario": scenario,
                    "thread_id": thread_id,
                    "test_complete": False,
                    "conversation_turns": 0,
                    "max_turns": 10,
                    "actual_tools_used": [],
                }

                # Run the evaluation
                config = {"configurable": {"thread_id": thread_id}}

                # Create a container for streaming updates
                stream_container = st.container()

                try:
                    # Run the graph
                    final_state = asyncio.run(
                        run_evaluation_async(
                            st.session_state.eval_graph,
                            initial_state,
                            config,
                            stream_container,
                        )
                    )

                    # Store results
                    st.session_state.test_results.append(
                        {
                            "timestamp": datetime.now(),
                            "scenario": scenario,
                            "tone": tone,
                            "score": final_state.get("evaluation_score", 0),  # type: ignore
                            "state": final_state,
                        }
                    )

                    st.success("✅ Evaluation completed!")

                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")

    with col2:
        st.header("Recent Results")

        if st.session_state.test_results:
            for i, result in enumerate(reversed(st.session_state.test_results[-5:])):
                with st.expander(
                    f"{result['scenario']} - Score: {result['score']}/10",
                    expanded=(i == 0),
                ):
                    st.write(
                        f"**Timestamp**: {result['timestamp'].strftime('%H:%M:%S')}"
                    )
                    st.write(f"**Advisor Type**: {result['tone']}")
                    st.write(f"**Evaluation Score**: {result['score']}/10")

                    if result["state"].get("evaluation_criteria"):
                        st.write("**Issues**:", result["state"]["evaluation_criteria"])
        else:
            st.info("No evaluation results yet. Run a test to see results.")

    # Results display
    if st.session_state.test_results:
        latest_result = st.session_state.test_results[-1]

        st.markdown("---")
        st.header("Latest Evaluation Details")

        # Conversation history
        with st.expander("📝 Conversation History", expanded=True):
            st.text(
                latest_result["state"].get("chatbot_history", "No conversation history")
            )

        # Evaluation metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score", f"{latest_result['score']}/10")
        with col2:
            st.metric("Turns", latest_result["state"].get("conversation_turns", 0))
        with col3:
            expected_tools = len(latest_result["state"].get("expected_tools_used", []))
            actual_tools = len(latest_result["state"].get("actual_tools_used", []))
            st.metric("Tools Used", f"{actual_tools}/{expected_tools}")
        with col4:
            st.metric(
                "Optimization",
                (
                    "Needed"
                    if latest_result["state"].get("optimization_needed")
                    else "Not Needed"
                ),
            )

        # Evaluation details
        with st.expander("📊 Evaluation Analysis"):
            st.write(
                "**Criteria:**",
                latest_result["state"].get("evaluation_criteria", "N/A"),
            )
            st.write(
                "**Evidence:**",
                latest_result["state"].get("evaluation_evidence", "N/A"),
            )

        # Prompt optimization
        if latest_result["state"].get("optimization_needed"):
            with st.expander("🔧 Prompt Optimization"):
                if latest_result["state"].get("optimized_prompt"):
                    st.subheader("Suggested Improvements")

                    # Show diff
                    original = latest_result["state"].get("original_prompt", "")
                    optimized = latest_result["state"].get("optimized_prompt", "")

                    if original and optimized:
                        diff_html = get_diff_html(original, optimized)
                        st.markdown(diff_html, unsafe_allow_html=True)

                    # Apply optimization button
                    if st.button("Apply Optimization"):
                        st.info(
                            "Optimization would be applied to the agent configuration"
                        )


async def run_evaluation_async(graph, initial_state, config, container):
    """Run the evaluation asynchronously with streaming updates"""
    final_state = None

    async for event in graph.astream(initial_state, config):
        # Update UI with streaming events
        for node, state in event.items():
            if node == "portfolio_agent":
                container.info(f"🤖 Agent: {state.get('message', 'Processing...')}")
            elif node == "simulate_advisor":
                if state.get("messages"):
                    container.success(f"👤 Advisor: {state['messages'][-1].content}")
            elif node == "evaluate":
                container.warning(
                    f"📊 Evaluation Score: {state.get('evaluation_score', 'Calculating...')}/10"
                )

        final_state = state

    return final_state


if __name__ == "__main__":
    main()
