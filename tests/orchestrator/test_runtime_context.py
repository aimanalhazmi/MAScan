from typing import Any

from mascan.contracts.reports import AgentQualityReview, AgentReport
from mascan.orchestrator.adapters import make_agent_subgraph
from mascan.orchestrator.state import GraphState, RuntimeContext


def test_outer_graph_state_excludes_agent_subgraph_bookkeeping() -> None:
    state_fields = set(GraphState.model_fields)

    assert "quality_reviews" not in state_fields
    assert "retry_feedback" not in state_fields
    assert "retry_attempts" not in state_fields


class RecordingAgent:
    name = "economics"

    def __init__(self) -> None:
        self.context: dict[str, Any] | None = None

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        self.context = context
        return AgentReport(
            agent_name=self.name,
            tasks=tasks,
            findings="agent report",
            sources=[],
            confidence=0.7,
            rendered_markdown="agent report",
        )


def test_agent_subgraph_passes_runtime_context_to_agent() -> None:
    agent = RecordingAgent()
    node = make_agent_subgraph(
        agent,  # type: ignore[arg-type]
        reviewer=lambda **kwargs: AgentQualityReview(
            agent_name=kwargs["agent_name"],
            status="sufficient",
            feedback="ok",
        ),
    )
    state = GraphState(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze AAPL"]},
        runtime_context=RuntimeContext(
            current_date="2026-06-02",
            timezone="Europe/Berlin",
        ),
    )

    result = node.invoke(state)

    assert result["reports"]["economics"].findings == "agent report"
    assert agent.context == {
        "user_input": "Analyze AAPL",
        "runtime": {
            "current_date": "2026-06-02",
            "timezone": "Europe/Berlin",
        },
    }
