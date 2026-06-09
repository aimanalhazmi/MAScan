from typing import Any

from mascan.agents.environmental.agent import EnvironmentalAgent
from mascan.agents.registry import agent_registry
from mascan.contracts.reports import AgentReport


def test_environmental_agent_is_registered() -> None:
    import mascan.agents.environmental  # noqa: F401

    assert "environmental" in agent_registry.all_names()


def test_environmental_agent_run_returns_report(mocker: Any) -> None:
    agent = EnvironmentalAgent()

    fake_result = {
        "messages": [
            type(
                "AIMessage",
                (),
                {"content": "Environmental risk findings", "tool_calls": []},
            )()
        ]
    }
    fake_agent = mocker.Mock()
    fake_agent.invoke.return_value = fake_result

    mocker.patch("mascan.agents.environmental.agent.get_chat_model")
    mocker.patch(
        "mascan.agents.environmental.agent.create_agent",
        return_value=fake_agent,
    )

    report = agent.run(tasks=["water stress in German manufacturing regions"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "environmental"
    assert report.tasks == ["water stress in German manufacturing regions"]
    assert report.findings == "Environmental risk findings"
    assert report.metadata["mode"] == "B — LLM-driven"
    assert report.metadata["deterministic_tools"] == []
    assert report.metadata["llm_chosen_tools"] == []
    assert "## Environmental Analysis" in report.rendered_markdown
    fake_agent.invoke.assert_called_once()
