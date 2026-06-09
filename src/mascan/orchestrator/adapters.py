from collections.abc import Callable
from typing import Any

from mascan.agents.base import BaseAgent
from mascan.core.logging import get_logger
from mascan.orchestrator.state import GraphState

logger = get_logger("orchestrator.adapters")


def make_agent_node(agent: BaseAgent) -> Callable[[GraphState], dict[str, Any]]:
    def node(state: GraphState) -> dict[str, Any]:
        assignment = state.plan.get(agent.name)
        if assignment is None or not assignment.tasks:
            logger.info("Agent %r has no tasks; skipping.", agent.name)
            return {}

        try:
            report = agent.run(
                tasks=assignment.tasks,
                context={
                    "objective_context": assignment.objective_context,
                    # Agents do not receive GraphState directly, so pass runtime metadata here.
                    "runtime": state.runtime_context.model_dump(),
                },
            )
            return {"reports": {agent.name: report}}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent %r failed", agent.name)
            return {"failures": {agent.name: f"{type(exc).__name__}: {exc}"}}

    node.__name__ = f"agent_{agent.name}"
    return node
