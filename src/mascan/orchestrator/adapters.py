from typing import Any, Callable

from mascan.agents.base import BaseAgent
from mascan.core.logging import get_logger
from mascan.orchestrator.state import GraphState

logger = get_logger("orchestrator.adapters")


def make_agent_node(agent: BaseAgent) -> Callable[[GraphState], dict[str, Any]]:
    def node(state: GraphState) -> dict[str, Any]:
        tasks = state.plan.get(agent.name, [])
        if not tasks:
            logger.info("Agent %r has no tasks; skipping.", agent.name)
            return {}

        try:
            report = agent.run(tasks=tasks, context={"user_input": state.user_input})
            return {"reports": {agent.name: report}}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent %r failed", agent.name)
            return {"failures": {agent.name: f"{type(exc).__name__}: {exc}"}}

    node.__name__ = f"agent_{agent.name}"
    return node