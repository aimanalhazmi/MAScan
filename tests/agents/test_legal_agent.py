from typing import Any

from mascan.agents.legal.agent import LegalAgent
from mascan.agents.registry import agent_registry
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult


def test_legal_agent_is_registered() -> None:
    import mascan.agents.legal  # noqa: F401

    assert "legal" in agent_registry.all_names()


def test_legal_agent_exposes_eur_lex_as_optional_tool() -> None:
    agent = LegalAgent()

    from mascan.agents.legal.agent import ALWAYS_CALL_TOOLS

    assert "eur_lex" in agent.tools
    optional_names = {tool.name for tool in agent.get_optional_tools()}
    assert "eur_lex" in optional_names
    assert "eur_lex" not in ALWAYS_CALL_TOOLS


def test_legal_agent_run_returns_report(mocker: Any) -> None:
    agent = LegalAgent()
    deterministic_outputs = {
        "federal_register": ToolResult(
            success=True,
            data={"query": "data privacy", "total_count": 1, "documents": []},
            source="federal_register:data privacy",
            metadata={"provider": "federalregister.gov", "returned": 1},
        )
    }

    mocker.patch.object(
        agent,
        "gather_deterministic",
        return_value=deterministic_outputs,
    )
    mocker.patch.object(
        agent,
        "run_react_agent",
        return_value=("Legal risk findings", []),
    )

    report = agent.run(tasks=["data privacy"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "legal"
    assert report.tasks == ["data privacy"]
    assert report.findings == "Legal risk findings"
    assert report.metadata["mode"] == "mixed"
    assert report.metadata["deterministic_tools"] == ["web_search", "federal_register"]
    assert "## Legal Analysis" in report.rendered_markdown
    assert [source.name for source in report.sources] == ["federal_register:data privacy"]
    assert report.sources[0].metadata == {"provider": "federalregister.gov", "returned": 1}
