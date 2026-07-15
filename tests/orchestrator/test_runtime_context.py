from typing import Any

from mascan.contracts import AgentAssignment
from mascan.contracts.metrics import ComponentMetrics
from mascan.contracts.reports import AgentReport
from mascan.orchestrator.adapters import make_agent_node
from mascan.orchestrator.state import GraphState, RuntimeContext


class RecordingAgent:
    name = "economics"

    def __init__(self) -> None:
        self.context: dict[str, Any] | None = None
        self.report = AgentReport(
            agent_name=self.name,
            tasks=["Analyze AAPL"],
            findings="agent report",
            rendered_markdown="agent report",
            component_metrics={
                self.name: ComponentMetrics(run_count=1, duration_seconds=1.25)
            },
        )

    def run(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
    ) -> AgentReport:
        self.context = context
        return self.report


def test_agent_node_passes_runtime_context_to_agent() -> None:
    agent = RecordingAgent()
    node = make_agent_node(agent)  # type: ignore[arg-type]
    state = GraphState(
        user_input="Analyze AAPL",
        plan={
            "economics": AgentAssignment(
                agent_name="economics",
                objective_context=(
                    "Assess the economic forces relevant to AAPL while preserving "
                    "the user's company-specific focus."
                ),
                tasks=["Analyze AAPL"],
            )
        },
        runtime_context=RuntimeContext(
            current_date="2026-06-02",
            timezone="Europe/Berlin",
        ),
    )

    result = node(state)

    assert result == {
        "reports": {"economics": agent.report},
        "component_metrics": agent.report.component_metrics,
    }
    assert agent.context == {
        "objective_context": (
            "Assess the economic forces relevant to AAPL while preserving "
            "the user's company-specific focus."
        ),
        "runtime": {
            "current_date": "2026-06-02",
            "timezone": "Europe/Berlin",
        },
    }
