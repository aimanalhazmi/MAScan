from mascan.contracts.reports import AgentReport
from mascan.orchestrator.state import GraphState
from mascan.orchestrator.synthesizer import _build_synthesis_prompt


def test_synthesis_prompt_includes_quality_review_notes() -> None:
    state = GraphState(
        user_input="Analyze AAPL",
        reports={
            "economics": AgentReport(
                agent_name="economics",
                tasks=["Analyze AAPL"],
                findings="Inflation exposure covered.",
                sources=[],
                confidence=0.5,
                rendered_markdown="Inflation exposure covered.",
                metadata={
                    "quality_review": {
                        "agent_name": "economics",
                        "status": "missing",
                        "feedback": "Reached retry limit with exchange-rate exposure still weak.",
                        "metadata": {},
                    }
                },
            )
        },
    )

    prompt = _build_synthesis_prompt(state)

    assert "Quality review notes:" in prompt
    assert "economics: missing" in prompt
    assert "exchange-rate exposure still weak" in prompt
