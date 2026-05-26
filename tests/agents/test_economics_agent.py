from typing import Any

from mascan.agents.economics.agent import EconomicsAgent
from mascan.agents.economics.prompts import build_user_prompt
from mascan.agents.economics.tools.market_data import WeeklyStockPricesTool
from mascan.agents.registry import agent_registry
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult


def test_economics_agent_is_registered() -> None:
    import mascan.agents.economics  # noqa: F401

    assert "economics" in agent_registry.all_names()


def test_economics_agent_loads_finance_and_web_tools() -> None:
    agent = EconomicsAgent()

    assert "web_query" in agent.tools
    assert "get_weekly_stock_prices" in agent.tools


def test_economics_prompt_guides_stock_tool_usage() -> None:
    prompt = build_user_prompt(
        tasks=["How could inflation and interest rates affect AAPL?"],
        tool_block="### Tool: web_query (source: web_query:firecrawl)\nMarket context",
    )

    assert "Use get_weekly_stock_prices when the task mentions" in prompt
    assert "ticker" in prompt
    assert "last 12 months" in prompt
    assert "2025-05-26" in prompt
    assert "2026-05-26" in prompt


def test_weekly_stock_tool_description_mentions_when_to_use_it() -> None:
    tool = WeeklyStockPricesTool()

    assert "public company" in tool.description
    assert "ticker" in tool.description
    assert "stock performance" in tool.description


def test_economics_agent_run_returns_report(mocker: Any) -> None:
    agent = EconomicsAgent()
    deterministic_outputs = {
        "web_query": ToolResult(
            success=True,
            data="Search summary for EU manufacturing outlook",
            source="web_query:firecrawl",
            metadata={"query": "EU manufacturing outlook"},
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
        return_value=("Economic outlook findings", []),
    )

    report = agent.run(tasks=["EU manufacturing outlook"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "economics"
    assert report.tasks == ["EU manufacturing outlook"]
    assert report.findings == "Economic outlook findings"
    assert report.metadata["mode"] == "mixed"
    assert report.metadata["deterministic_tools"] == ["web_query"]
    assert "## Economics Analysis" in report.rendered_markdown
    assert [source.name for source in report.sources] == ["web_query:firecrawl"]
    assert report.sources[0].metadata == {"query": "EU manufacturing outlook"}
