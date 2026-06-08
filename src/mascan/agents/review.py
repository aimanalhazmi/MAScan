from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from mascan.agents.retry import build_retry_feedback
from mascan.contracts.reports import AgentQualityReview, AgentReport
from mascan.core.llm import get_chat_model
from mascan.core.settings import get_settings


class AgentReviewer(Protocol):
    def __call__(
        self,
        *,
        agent_name: str,
        tasks: list[str],
        report: AgentReport,
        state: Any,
    ) -> AgentQualityReview: ...


REVIEWER_SYSTEM_PROMPT = """\
You are a quality gate for one domain agent in a PESTEL multi-agent system.

Review only the given agent report against its assigned tasks.

Return:
- sufficient: the report directly answers all assigned tasks with enough useful detail.
- missing: the report is useful but incomplete; specify the missing gaps.
- failed: the report is unusable, off-topic, empty, or too unreliable to build on.

Be strict but practical. Feedback must be specific enough for the same agent to retry.
"""


def review_agent_report(
    *,
    agent_name: str,
    tasks: list[str],
    report: AgentReport,
    state: Any,
) -> AgentQualityReview:
    """Review one agent report with structured LLM output."""
    settings = get_settings()
    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.0,
        max_tokens=800,
    )
    structured_llm = llm.with_structured_output(
        AgentQualityReview,
        method="function_calling",
    )
    review = cast(
        AgentQualityReview,
        structured_llm.invoke(
            [
                SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                HumanMessage(content=build_review_prompt(agent_name, tasks, report, state)),
            ],
        ),
    )
    if review.agent_name != agent_name:
        review = review.model_copy(update={"agent_name": agent_name})
    return review


def make_review_agent_node(
    agent_name: str,
    reviewer: AgentReviewer | None = None,
) -> Callable[[Any], dict[str, Any]]:
    """Build a LangGraph node that reviews one agent's latest report."""
    review_report = reviewer if reviewer is not None else review_agent_report

    def review_agent(state: Any) -> dict[str, Any]:
        tasks = state.plan.get(agent_name, [])
        current_review = state.quality_review
        if current_review is not None and current_review.metadata.get("runtime_failure") is True:
            return {}

        report = state.reports.get(agent_name)
        if not tasks or report is None:
            return {}

        review = review_report(
            agent_name=agent_name,
            tasks=tasks,
            report=report,
            state=state,
        )
        attempts = state.retry_attempts
        should_retry = review.status != "sufficient" and attempts < state.max_iterations
        review = review.model_copy(
            update={
                "metadata": {
                    **review.metadata,
                    "retry": should_retry,
                    "attempt": attempts,
                }
            }
        )

        update: dict[str, Any] = {
            "quality_review": review,
            "reports": {
                agent_name: report.model_copy(
                    update={
                        "metadata": {
                            **report.metadata,
                            "quality_review": review.model_dump(mode="json"),
                        }
                    }
                )
            },
        }
        if should_retry:
            update["retry_attempts"] = attempts + 1
            update["retry_feedback"] = build_retry_feedback(review=review, report=report)
        return update

    return review_agent


def route_after_review(agent_name: str, state: Any) -> str:
    """Route one agent subgraph after the review node."""
    review = state.quality_review
    if review is not None and review.metadata.get("retry") is True:
        return "run_agent"
    return END


def build_review_prompt(
    agent_name: str,
    tasks: list[str],
    report: AgentReport,
    state: Any,
) -> str:
    task_lines = "\n".join(f"- {task}" for task in tasks)
    sources = ", ".join(source.name for source in report.sources) or "(none)"
    return (
        f"User question:\n{state.user_input}\n\n"
        f"Agent under review: {agent_name}\n\n"
        f"Assigned tasks:\n{task_lines}\n\n"
        f"Agent confidence: {report.confidence:.2f}\n"
        f"Sources: {sources}\n\n"
        f"Agent findings:\n{report.findings}\n"
    )
