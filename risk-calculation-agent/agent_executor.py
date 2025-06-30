from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from agent import initialize_agent, extract_response_content
from typing import override
import logging


class RiskCalculationAgentExecutor(AgentExecutor):
    """Custom executor for the Risk Calculation Agent."""

    def __init__(self):
        self.agent, self.config = initialize_agent()

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = {"messages": [{"role": "user", "content": context.get_user_input()}]}
        result = await self.agent.ainvoke(query, config=self.config)  # type: ignore
        await event_queue.enqueue_event(
            new_agent_text_message(extract_response_content(result))
        )  # type: ignore

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")
