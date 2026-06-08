from __future__ import annotations

from typing import Any

from mascan.contracts.reports import AgentQualityReview, AgentReport, AgentRetryFeedback


def retry_context_from_feedback(feedback: AgentRetryFeedback | None) -> dict[str, Any]:
    if feedback is None:
        return {}
    return {"retry_feedback": feedback.model_dump(exclude_none=True)}


def build_retry_feedback(
    *,
    review: AgentQualityReview,
    report: AgentReport,
) -> AgentRetryFeedback:
    if review.status == "missing":
        return AgentRetryFeedback(
            status="missing",
            feedback=review.feedback,
            previous_report=report.findings,
            instruction=(
                "Use the previous report as a base and return a complete revised report "
                "that fills only the missing gaps identified by the quality feedback."
            ),
        )
    return AgentRetryFeedback(
        status="failed",
        feedback=review.feedback,
        instruction=(
            "Redo the original task from scratch using only this quality feedback. "
            "Do not rely on the previous failed report."
        ),
    )
