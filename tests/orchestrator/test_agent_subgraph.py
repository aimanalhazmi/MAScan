from types import SimpleNamespace
from typing import Any

from langgraph.graph import END, START, StateGraph

from mascan.agents import review as review_module
from mascan.contracts.reports import AgentQualityReview, AgentReport
from mascan.orchestrator.adapters import make_agent_subgraph
from mascan.orchestrator.state import GraphState, RuntimeContext


def quality_review(result: dict[str, Any], agent_name: str = "economics") -> dict[str, Any]:
    return result["reports"][agent_name].metadata["quality_review"]


class SequenceAgent:
    def __init__(self, findings: list[str], name: str = "economics") -> None:
        self.name = name
        self.findings = findings
        self.calls = 0
        self.contexts: list[dict[str, Any] | None] = []

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        self.contexts.append(context)
        finding = self.findings[min(self.calls, len(self.findings) - 1)]
        self.calls += 1
        return AgentReport(
            agent_name=self.name,
            tasks=tasks,
            findings=finding,
            sources=[],
            confidence=0.7,
            rendered_markdown=f"## Economics\n\n{finding}",
        )


class FailsOnRetryAgent:
    name = "economics"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("retry failed")
        return AgentReport(
            agent_name=self.name,
            tasks=tasks,
            findings="incomplete answer",
            sources=[],
            confidence=0.4,
            rendered_markdown="incomplete answer",
        )


def test_agent_subgraph_accepts_sufficient_report() -> None:
    agent = SequenceAgent(["complete answer"])

    def reviewer(
        *,
        agent_name: str,
        tasks: list[str],
        report: AgentReport,
        state: GraphState,
    ) -> AgentQualityReview:
        return AgentQualityReview(
            agent_name=agent_name,
            status="sufficient",
            feedback="Report covers the assigned task.",
        )

    subgraph = make_agent_subgraph(agent, reviewer=reviewer)  # type: ignore[arg-type]
    state = GraphState(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze AAPL"]},
        runtime_context=RuntimeContext(current_date="2026-06-09", timezone="Europe/Berlin"),
    )

    result = subgraph.invoke(state)

    assert agent.calls == 1
    assert result["reports"]["economics"].findings == "complete answer"
    assert "quality_reviews" not in result
    assert quality_review(result)["status"] == "sufficient"
    assert agent.contexts[0] == {
        "user_input": "Analyze AAPL",
        "runtime": {
            "current_date": "2026-06-09",
            "timezone": "Europe/Berlin",
        },
    }


def test_agent_subgraph_retries_missing_report_with_previous_report_and_feedback() -> None:
    agent = SequenceAgent(["incomplete answer", "complete revised answer"])
    review_statuses = ["missing", "sufficient"]

    def reviewer(
        *,
        agent_name: str,
        tasks: list[str],
        report: AgentReport,
        state: GraphState,
    ) -> AgentQualityReview:
        status = review_statuses.pop(0)
        return AgentQualityReview(
            agent_name=agent_name,
            status=status,  # type: ignore[arg-type]
            feedback="Add the missing exchange-rate exposure.",
        )

    subgraph = make_agent_subgraph(agent, reviewer=reviewer)  # type: ignore[arg-type]
    state = GraphState(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze AAPL"]},
        runtime_context=RuntimeContext(current_date="2026-06-09", timezone="Europe/Berlin"),
    )

    result = subgraph.invoke(state)

    assert agent.calls == 2
    assert result["reports"]["economics"].findings == "complete revised answer"
    assert "quality_reviews" not in result
    assert quality_review(result)["status"] == "sufficient"
    assert agent.contexts[1]["retry_feedback"] == {
        "status": "missing",
        "feedback": "Add the missing exchange-rate exposure.",
        "previous_report": "incomplete answer",
        "instruction": (
            "Use the previous report as a base and return a complete revised report "
            "that fills only the missing gaps identified by the quality feedback."
        ),
    }


def test_agent_subgraph_retries_failed_report_with_feedback_only() -> None:
    agent = SequenceAgent(["off-topic answer", "redone complete answer"])
    review_statuses = ["failed", "sufficient"]

    def reviewer(
        *,
        agent_name: str,
        tasks: list[str],
        report: AgentReport,
        state: GraphState,
    ) -> AgentQualityReview:
        status = review_statuses.pop(0)
        return AgentQualityReview(
            agent_name=agent_name,
            status=status,  # type: ignore[arg-type]
            feedback="Previous answer was off-topic and lacked sources.",
        )

    subgraph = make_agent_subgraph(agent, reviewer=reviewer)  # type: ignore[arg-type]
    state = GraphState(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze AAPL"]},
        runtime_context=RuntimeContext(current_date="2026-06-09", timezone="Europe/Berlin"),
    )

    result = subgraph.invoke(state)

    assert agent.calls == 2
    assert result["reports"]["economics"].findings == "redone complete answer"
    assert "quality_reviews" not in result
    assert quality_review(result)["status"] == "sufficient"
    assert agent.contexts[1]["retry_feedback"] == {
        "status": "failed",
        "feedback": "Previous answer was off-topic and lacked sources.",
        "instruction": (
            "Redo the original task from scratch using only this quality feedback. "
            "Do not rely on the previous failed report."
        ),
    }


def test_agent_subgraph_stops_retrying_when_max_iterations_is_reached() -> None:
    agent = SequenceAgent(["first incomplete answer", "second incomplete answer", "unused"])

    def reviewer(
        *,
        agent_name: str,
        tasks: list[str],
        report: AgentReport,
        state: GraphState,
    ) -> AgentQualityReview:
        return AgentQualityReview(
            agent_name=agent_name,
            status="missing",
            feedback="Still missing exchange-rate exposure.",
        )

    subgraph = make_agent_subgraph(agent, reviewer=reviewer)  # type: ignore[arg-type]
    state = GraphState(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze AAPL"]},
        runtime_context=RuntimeContext(current_date="2026-06-09", timezone="Europe/Berlin"),
        max_iterations=1,
    )

    result = subgraph.invoke(state)

    assert agent.calls == 2
    assert result["reports"]["economics"].findings == "second incomplete answer"
    assert "quality_reviews" not in result
    assert quality_review(result)["status"] == "missing"
    assert quality_review(result)["metadata"]["retry"] is False


def test_agent_subgraph_stops_when_retry_execution_fails() -> None:
    agent = FailsOnRetryAgent()

    def reviewer(
        *,
        agent_name: str,
        tasks: list[str],
        report: AgentReport,
        state: GraphState,
    ) -> AgentQualityReview:
        return AgentQualityReview(
            agent_name=agent_name,
            status="missing",
            feedback="Add the missing exchange-rate exposure.",
        )

    subgraph = make_agent_subgraph(agent, reviewer=reviewer)  # type: ignore[arg-type]
    state = GraphState(
        user_input="Analyze AAPL",
        plan={"economics": ["Analyze AAPL"]},
        runtime_context=RuntimeContext(current_date="2026-06-09", timezone="Europe/Berlin"),
    )

    result = subgraph.invoke(state)

    assert agent.calls == 2
    assert result["reports"]["economics"].findings == "incomplete answer"
    assert result["failures"]["economics"] == "RuntimeError: retry failed"
    assert "quality_reviews" not in result
    assert quality_review(result)["status"] == "failed"
    assert quality_review(result)["metadata"]["runtime_failure"] is True
    assert quality_review(result)["metadata"]["retry"] is False


def test_default_reviewer_uses_structured_llm_output(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeStructuredLLM:
        def invoke(self, messages: list[Any]) -> AgentQualityReview:
            captured["messages"] = messages
            return AgentQualityReview(
                agent_name="economics",
                status="missing",
                feedback="The report did not address demand exposure.",
            )

    class FakeLLM:
        def with_structured_output(
            self,
            model: type[AgentQualityReview],
            method: str,
        ) -> FakeStructuredLLM:
            captured["model"] = model
            captured["method"] = method
            return FakeStructuredLLM()

    monkeypatch.setattr(
        review_module,
        "get_settings",
        lambda: SimpleNamespace(openai_model_default="test-model"),
    )
    monkeypatch.setattr(review_module, "get_chat_model", lambda **_: FakeLLM())

    report = AgentReport(
        agent_name="economics",
        tasks=["Analyze AAPL"],
        findings="Inflation only.",
        sources=[],
        confidence=0.4,
        rendered_markdown="Inflation only.",
    )
    state = GraphState(user_input="Analyze AAPL", plan={"economics": ["Analyze AAPL"]})

    review = review_module.review_agent_report(
        agent_name="economics",
        tasks=["Analyze AAPL"],
        report=report,
        state=state,
    )

    assert review.status == "missing"
    assert review.feedback == "The report did not address demand exposure."
    assert captured["model"] is AgentQualityReview
    assert captured["method"] == "function_calling"
    assert "Analyze AAPL" in captured["messages"][1].content
    assert "Inflation only." in captured["messages"][1].content


def test_outer_graph_can_run_agent_subgraphs_in_parallel() -> None:
    economics = SequenceAgent(["economics answer"], name="economics")
    political = SequenceAgent(["political answer"], name="political")

    def reviewer(
        *,
        agent_name: str,
        tasks: list[str],
        report: AgentReport,
        state: GraphState,
    ) -> AgentQualityReview:
        return AgentQualityReview(
            agent_name=agent_name,
            status="sufficient",
            feedback="ok",
        )

    graph = StateGraph(GraphState)
    graph.add_node(
        "planner",
        lambda _: {
            "plan": {
                "economics": ["Analyze economics"],
                "political": ["Analyze politics"],
            }
        },
    )
    graph.add_node("economics", make_agent_subgraph(economics, reviewer=reviewer))  # type: ignore[arg-type]
    graph.add_node("political", make_agent_subgraph(political, reviewer=reviewer))  # type: ignore[arg-type]
    graph.add_node("done", lambda _: {})
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "economics")
    graph.add_edge("planner", "political")
    graph.add_edge("economics", "done")
    graph.add_edge("political", "done")
    graph.add_edge("done", END)

    result = graph.compile().invoke(GraphState(user_input="Analyze market"))

    assert result["reports"]["economics"].findings == "economics answer"
    assert result["reports"]["political"].findings == "political answer"
    assert "quality_reviews" not in result
    assert quality_review(result, "economics")["status"] == "sufficient"
    assert quality_review(result, "political")["status"] == "sufficient"
