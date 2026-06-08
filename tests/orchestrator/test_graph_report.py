from mascan.contracts.reports import AgentReport
from mascan.orchestrator.graph import state_to_report


def test_state_to_report_includes_quality_reviews_in_metadata() -> None:
    report = state_to_report(
        {
            "user_input": "Analyze AAPL",
            "final_summary": "Summary",
            "final_markdown": "Markdown",
            "plan": {"economics": ["Analyze AAPL"]},
            "reports": {
                "economics": AgentReport(
                    agent_name="economics",
                    tasks=["Analyze AAPL"],
                    findings="Findings",
                    sources=[],
                    confidence=0.7,
                    rendered_markdown="Findings",
                    metadata={
                        "quality_review": {
                            "agent_name": "economics",
                            "status": "sufficient",
                            "feedback": "ok",
                            "metadata": {},
                        }
                    },
                )
            },
            "failures": {},
        }
    )

    assert report.metadata["quality_reviews"]["economics"]["status"] == "sufficient"
    assert report.metadata["quality_reviews"]["economics"]["feedback"] == "ok"
