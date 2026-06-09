from typing import Any

from mascan.agents.political.agent import PoliticalAgent
from mascan.agents.registry import agent_registry
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult


def test_political_agent_is_registered() -> None:
    import mascan.agents.political  # noqa: F401

    assert "political" in agent_registry.all_names()


def test_political_agent_run_returns_report(mocker: Any) -> None:
    agent = PoliticalAgent()
    deterministic_outputs = {
        "web_search": ToolResult(
            success=True,
            data=[{"title": "Policy update", "url": "https://example.com"}],
            source="web_search:firecrawl",
            metadata={"query": "EU battery regulation", "count": 1},
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
        return_value=("Political risk findings", []),
    )

    report = agent.run(tasks=["EU battery regulation"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "political"
    assert report.tasks == ["EU battery regulation"]
    assert report.findings == "Political risk findings"
    assert report.metadata["mode"] == "mixed"
    assert report.metadata["deterministic_tools"] == ["web_search", "news_api"]
    assert "## Political Analysis" in report.rendered_markdown
    # Sources are real article links labelled by title, not tool names.
    assert [source.url for source in report.sources] == ["https://example.com"]
    assert report.sources[0].name == "Policy update"
    assert "[Policy update](https://example.com)" in report.rendered_markdown
