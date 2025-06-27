from agent_executor import RiskCalculationAgentExecutor
from agent import get_mcp_tools
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from typing import List


def convert_mcp_tools_to_a2a_skills(tools_list: List) -> List[AgentSkill]:
    """
    Convert a list of MCP StructuredTool objects to A2A AgentSkill objects.

    Args:
        tools_list: List of StructuredTool objects from MCP

    Returns:
        List[AgentSkill]: List of AgentSkill objects
    """
    if not tools_list:
        return []

    skills = []

    for tool in tools_list:
        skill = AgentSkill(
            id=tool.name,
            name=tool.name.replace("_", " ").title(),
            description=(
                _extract_main_description(tool.description)
                if hasattr(tool, "description")
                else ""
            ),
            tags=[],
        )

        skills.append(skill)

    return skills


def _extract_main_description(description: str) -> str:
    """Extract the main description from tool description, removing Args/Returns sections."""
    if not description:
        return ""

    # Split description into lines and clean up
    lines = [line.strip() for line in description.split("\n") if line.strip()]

    if not lines:
        return ""

    # Find the first line that doesn't contain Args: or Returns:
    main_desc_lines = []
    for line in lines:
        if line.startswith("Args:") or line.startswith("Returns:"):
            break
        main_desc_lines.append(line)

    return " ".join(main_desc_lines).strip()


if __name__ == "__main__":
    mcp_tools = get_mcp_tools()
    a2a_skills = convert_mcp_tools_to_a2a_skills(mcp_tools)

    agent_card = AgentCard(
        name="Risk Calculation Agent",
        description="An agent that calculates risk (e.g., VaR, CVaR, etc) for a portfolio.",
        url="http://localhost:9999",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=a2a_skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=RiskCalculationAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)

    import uvicorn

    uvicorn.run(server.build(), host="0.0.0.0", port=9999)
