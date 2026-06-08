from types import SimpleNamespace
from typing import Any

from langgraph.graph import END

from mascan.agents import review as review_module
from mascan.contracts.reports import AgentQualityReview, AgentReport


def test_review_node_marks_missing_report_for_retry() -> None:
    def reviewer(
        *,
        agent_name: str,
        tasks: list[str],
        report: AgentReport,
            state: Any,
    ) -> AgentQualityReview:
        return AgentQualityReview(
            agent_name=agent_name,
            status="missing",
            feedback="Add exchange-rate exposure.",
        )

    report = AgentReport(
        agent_name="economics",
        tasks=["Analyze economics"],
        findings="Inflation only.",
        sources=[],
        confidence=0.5,
        rendered_markdown="Inflation only.",
    )
    state = SimpleNamespace(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze economics"]},
        reports={"economics": report},
        quality_review=None,
        retry_attempts=0,
        max_iterations=10,
    )

    node = review_module.make_review_agent_node("economics", reviewer=reviewer)
    update = node(state)

    review = update["quality_review"]
    assert review.status == "missing"
    assert review.metadata == {"retry": True, "attempt": 0}
    assert update["retry_attempts"] == 1
    assert update["retry_feedback"].previous_report == "Inflation only."
    assert update["reports"]["economics"].metadata["quality_review"]["status"] == "missing"


def test_route_after_review_retries_only_when_review_requests_retry() -> None:
    retry_state = SimpleNamespace(
        user_input="Analyze AAPL",
        quality_review=AgentQualityReview(
            agent_name="economics",
            status="missing",
            feedback="Add exchange-rate exposure.",
            metadata={"retry": True},
        ),
    )
    end_state = SimpleNamespace(
        user_input="Analyze AAPL",
        quality_review=AgentQualityReview(
            agent_name="economics",
            status="sufficient",
            feedback="ok",
            metadata={"retry": False},
        ),
    )

    assert review_module.route_after_review("economics", retry_state) == "run_agent"
    assert review_module.route_after_review("economics", end_state) == END


def test_review_node_skips_after_runtime_failure() -> None:
    state = SimpleNamespace(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze economics"]},
        reports={},
        quality_review=AgentQualityReview(
            agent_name="economics",
            status="failed",
            feedback="Agent execution failed.",
            metadata={"runtime_failure": True, "retry": False},
        ),
        retry_attempts=0,
        max_iterations=10,
    )
    reviewer_called: list[bool] = []

    def reviewer(**_: Any) -> AgentQualityReview:
        reviewer_called.append(True)
        return AgentQualityReview(agent_name="economics", status="sufficient", feedback="ok")

    node = review_module.make_review_agent_node("economics", reviewer=reviewer)

    assert node(state) == {}
    assert reviewer_called == []
