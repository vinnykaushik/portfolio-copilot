import logging
from agent import root_agent
from agent_executor import ADKAgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting A2A server for Portfolio Analytics Agent...")

    try:
        get_all_portfolios = AgentSkill(
            id="get_all_portfolios_tool",
            name="Get All Portfolios",
            description="Retrieves all portfolios for the user.",
            tags=["portfolio", "user"],
            examples=["Show my portfolios", "List all portfolios"],
        )
        get_portfolio_by_id = AgentSkill(
            id="get_portfolio_by_id_tool",
            name="Get Portfolio By ID",
            description="Retrieves a specific portfolio by its ID.",
            tags=["portfolio", "id"],
            examples=[
                "Get portfolio with ID 12345",
                "Show portfolio details for ID 67890",
            ],
        )
        get_portfolio_by_name = AgentSkill(
            id="get_portfolio_by_name_tool",
            name="Get Portfolio By Name",
            description="Retrieves a specific portfolio by the customer's name.",
            tags=["portfolio", "name"],
            examples=[
                "Get portfolio for John Doe",
                "Show portfolio details for Alice Smith",
            ],
        )
        a2a_skills = [get_all_portfolios, get_portfolio_by_id, get_portfolio_by_name]
        logger.info(f"Converted to {len(a2a_skills)} A2A skills")

        agent_card = AgentCard(
            name="Customer Portfolio Agent",
            description="An agent that retrieves and manages customer portfolio information. It can fetch all portfolios and retrieve specific portfolios by ID or name.",
            url="http://localhost:7777",
            version="1.0.0",
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=a2a_skills,
        )

        logger.info("Creating request handler...")
        request_handler = DefaultRequestHandler(
            agent_executor=ADKAgentExecutor(agent=root_agent),
            task_store=InMemoryTaskStore(),
        )

        logger.info("Creating A2A FastAPI application...")
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        logger.info("Starting server on http://0.0.0.0:7777...")
        import uvicorn

        uvicorn.run(server.build(), host="0.0.0.0", port=7777, log_level="info")

    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise
