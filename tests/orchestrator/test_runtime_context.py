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
        "user_input": "Analyze AAPL",
        "rag_evidence": [],
        "provided_sources": [],
        "runtime": {
            "current_date": "2026-06-02",
            "timezone": "Europe/Berlin",
        },
    }


def test_agent_node_passes_raw_upload_evidence_and_citable_file_source() -> None:
    agent = RecordingAgent()
    node = make_agent_node(agent)  # type: ignore[arg-type]
    evidence = [
        {
            "content": "Adjusted EBITDA reached EUR 522 million.",
            "citation": {
                "document": "EVONIK Q1 2026 Factsheet & Update.pdf",
                "page": 4,
            },
        },
        {
            "content": "Evidence from another upload.",
            "citation": {"document": "Unassigned.pdf", "page": 1},
        },
    ]
    state = GraphState(
        user_input="Use the attached EVONIK factsheet.",
        plan={
            "economics": AgentAssignment(
                agent_name="economics",
                objective_context="Assess EVONIK's financial position.",
                tasks=["Use the factsheet evidence"],
                evidence_documents=["EVONIK Q1 2026 Factsheet & Update.pdf"],
            )
        },
        rag_evidence=evidence,
    )

    node(state)

    assert agent.context is not None
    assert agent.context["rag_evidence"] == [evidence[0]]
    assert agent.context["user_input"] == state.user_input
    provided = agent.context["provided_sources"]
    assert len(provided) == 1
    assert provided[0].url == (
        "/rag/files/EVONIK%20Q1%202026%20Factsheet%20%26%20Update.pdf"
    )
    assert provided[0].metadata["citation"]["pages"] == [4]
    assert "Adjusted EBITDA" in provided[0].metadata["content"]
    assert "another upload" not in provided[0].metadata["content"]
