from typing import Any

from mascan.orchestrator.adapters import make_agent_node
from mascan.orchestrator.state import GraphState, RuntimeContext


class RecordingAgent:
    name = "economics"

    def __init__(self) -> None:
        self.context: dict[str, Any] | None = None

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> str:
        self.context = context
        return "agent report"


def test_agent_node_passes_runtime_context_to_agent() -> None:
    agent = RecordingAgent()
    node = make_agent_node(agent)  # type: ignore[arg-type]
    state = GraphState(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze AAPL"]},
        runtime_context=RuntimeContext(
            current_date="2026-06-02",
            timezone="Europe/Berlin",
        ),
    )

    result = node(state)

    assert result == {"reports": {"economics": "agent report"}}
    assert agent.context == {
        "user_input": "Analyze AAPL",
        "runtime": {
            "current_date": "2026-06-02",
            "timezone": "Europe/Berlin",
        },
    }
