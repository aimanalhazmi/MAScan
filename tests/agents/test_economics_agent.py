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

    assert "web_search" in agent.tools
    assert "get_weekly_stock_prices" in agent.tools


def test_economics_prompt_guides_stock_tool_usage() -> None:
    prompt = build_user_prompt(
        tasks=["How could inflation and interest rates affect AAPL?"],
        tool_block="### Tool: web_search (source: web_search:firecrawl)\nMarket context",
        context={
            "runtime": {
                "current_date": "2026-06-02",
                "timezone": "Europe/Berlin",
            }
        },
    )

    assert "Runtime context:" in prompt
    assert "Current date: 2026-06-02" in prompt
    assert "Timezone: Europe/Berlin" in prompt


def test_economics_prompt_renders_retry_feedback_with_previous_report() -> None:
    prompt = build_user_prompt(
        tasks=["Analyze economic risks for AAPL"],
        tool_block="### Tool: web_search (source: web_search:firecrawl)\nMarket context",
        context={
            "retry_feedback": {
                "status": "missing",
                "feedback": "The report missed exchange-rate exposure.",
                "previous_report": "Prior report covered inflation only.",
                "instruction": "Use the previous report as a base and return a complete revised report.",
            }
        },
    )

    assert "Quality gate retry feedback:" in prompt
    assert "Status: missing" in prompt
    assert "The report missed exchange-rate exposure." in prompt
    assert "Prior report covered inflation only." in prompt
    assert "Use the previous report as a base" in prompt


def test_weekly_stock_tool_description_mentions_when_to_use_it() -> None:
    tool = WeeklyStockPricesTool()

    assert "public company" in tool.description
    assert "ticker" in tool.description
    assert "stock performance" in tool.description


def test_economics_agent_run_returns_report(mocker: Any) -> None:
    agent = EconomicsAgent()
    deterministic_outputs = {
        "web_search": ToolResult(
            success=True,
            data="Search summary for EU manufacturing outlook",
            source="web_search:firecrawl",
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
    assert report.metadata["deterministic_tools"] == ["web_search"]
    assert "## Economics Analysis" in report.rendered_markdown
    assert [source.name for source in report.sources] == ["web_search:firecrawl"]
    assert report.sources[0].metadata == {"query": "EU manufacturing outlook"}
