from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from mascan.agents.base import BaseAgent
from mascan.agents.retry import retry_context_from_feedback
from mascan.agents.review import AgentReviewer, make_review_agent_node, route_after_review
from mascan.contracts.reports import AgentQualityReview, AgentReport, AgentRetryFeedback
from mascan.core.logging import get_logger
from mascan.orchestrator.state import RuntimeContext

logger = get_logger("orchestrator.adapters")


class AgentSubgraphOutput(BaseModel):
    """Public output from an agent subgraph to the outer graph."""

    reports: dict[str, AgentReport] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)


class AgentSubgraphState(BaseModel):
    """Private state for one agent subgraph."""

    user_input: str
    runtime_context: RuntimeContext = Field(default_factory=RuntimeContext.from_system)
    plan: dict[str, list[str]] = Field(default_factory=dict)
    reports: dict[str, AgentReport] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)
    max_iterations: int = 10

    quality_review: AgentQualityReview | None = None
    retry_feedback: AgentRetryFeedback | None = None
    retry_attempts: int = 0

    model_config = {"arbitrary_types_allowed": True}


def make_agent_subgraph(agent: BaseAgent, reviewer: AgentReviewer | None = None) -> Any:
    """Build a per-agent subgraph: run the agent, then review its report."""
    def run_agent(state: AgentSubgraphState) -> dict[str, Any]:
        tasks = state.plan.get(agent.name, [])
        if not tasks:
            logger.info("Agent %r has no tasks; skipping.", agent.name)
            return {}

        try:
            report = agent.run(
                tasks=tasks,
                context={
                    "user_input": state.user_input,
                    # Agents do not receive GraphState directly, so pass runtime metadata here.
                    "runtime": state.runtime_context.model_dump(),
                    **retry_context_from_feedback(state.retry_feedback),
                },
            )
            return {"reports": {agent.name: report}}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent %r failed", agent.name)
            failure = f"{type(exc).__name__}: {exc}"
            review = AgentQualityReview(
                agent_name=agent.name,
                status="failed",
                feedback=f"Agent execution failed: {failure}",
                metadata={
                    "runtime_failure": True,
                    "retry": False,
                    "attempt": state.retry_attempts,
                },
            )
            previous_report = state.reports.get(agent.name)
            update: dict[str, Any] = {
                "failures": {agent.name: failure},
                "quality_review": review,
            }
            if previous_report is not None:
                update["reports"] = {
                    agent.name: previous_report.model_copy(
                        update={
                            "metadata": {
                                **previous_report.metadata,
                                "quality_review": review.model_dump(mode="json"),
                            }
                        }
                    )
                }
            return {
                **update,
            }

    subgraph = StateGraph(AgentSubgraphState, output_schema=AgentSubgraphOutput)
    subgraph.add_node("run_agent", run_agent)
    subgraph.add_node(
        "review_agent",
        cast(Any, make_review_agent_node(agent.name, reviewer=reviewer)),
    )
    subgraph.add_edge(START, "run_agent")
    subgraph.add_edge("run_agent", "review_agent")
    subgraph.add_conditional_edges(
        "review_agent",
        lambda state: route_after_review(agent.name, state),
        {"run_agent": "run_agent", END: END},
    )
    return subgraph.compile()
