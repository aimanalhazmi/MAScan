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

    assert "eur_lex" in agent.tools
    optional_names = {tool.name for tool in agent.get_optional_tools()}
    assert "eur_lex" in optional_names
    assert "eur_lex" not in agent.config.always_call_tools


def test_legal_agent_run_returns_report(mocker: Any) -> None:
    agent = LegalAgent()
    fed_url = (
        "https://www.federalregister.gov/documents/2026/01/02/"
        "2026-00001/consumer-data-privacy-rule"
    )
    deterministic_outputs = {
        "federal_register": ToolResult(
            success=True,
            data={
                "query": "data privacy",
                "total_count": 1,
                "documents": [
                    {
                        "title": "Consumer Data Privacy Rule",
                        "document_number": "2026-00001",
                        "url": fed_url,
                    }
                ],
            },
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
        return_value=({"messages": []}, "Legal risk findings", []),
    )

    report = agent.run(tasks=["data privacy"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "legal"
    assert report.tasks == ["data privacy"]
    assert report.findings == "Legal risk findings"
    assert report.metadata["mode"] == "mixed"
    # No agent config sets always_call_tools anymore, so this metadata is empty.
    assert report.metadata["deterministic_tools"] == []
    assert "## Legal Analysis" in report.rendered_markdown
    # Sources are article-level links harvested from the tool output, labelled by
    # the document title, with a {"tool": ...} provenance tag.
    assert [source.url for source in report.sources] == [fed_url]
    assert report.sources[0].name == "Consumer Data Privacy Rule"
    assert report.sources[0].metadata["tool"] == "federal_register"
