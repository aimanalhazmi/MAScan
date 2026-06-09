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
    )

    assert "Use get_weekly_stock_prices when the task mentions" in prompt
    assert "ticker" in prompt
    assert "last 12 months relative to the runtime current date" in prompt


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
        return_value=("Economic outlook findings", [], []),
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


class _ToolMessage:
    """Minimal stand-in for a LangChain ToolMessage."""

    def __init__(self, name: str, content: Any) -> None:
        self.type = "tool"
        self.name = name
        self.content = content


def test_extract_llm_sources_surfaces_market_data() -> None:
    payload = {
        "ticker": "BMW.DE",
        "start_date": "2025-06-01",
        "end_date": "2026-06-01",
        "fundamentals": {"company_name": "Bayerische Motoren Werke AG"},
        "weekly_prices": [],
    }
    result = {
        "messages": [
            _ToolMessage("get_weekly_stock_prices", payload),
            _ToolMessage("get_weekly_stock_prices", payload),  # deduped
        ]
    }

    sources = EconomicsAgent.extract_llm_sources(result)

    assert len(sources) == 1
    source = sources[0]
    assert source.name == "yfinance:BMW.DE"
    assert source.url == "https://finance.yahoo.com/quote/BMW.DE"
    assert source.metadata["ticker"] == "BMW.DE"
    assert source.metadata["company_name"] == "Bayerische Motoren Werke AG"

    line = EconomicsAgent.format_source_line(source)
    assert line == "- [yfinance:BMW.DE](https://finance.yahoo.com/quote/BMW.DE)"


def test_weekly_stock_tool_fails_on_delisted_ticker(mocker: Any) -> None:
    import pandas as pd

    from mascan.agents.economics.tools import market_data

    fake_ticker = mocker.Mock()
    fake_ticker.info = {}
    fake_ticker.history.return_value = pd.DataFrame()  # delisted -> empty frame
    mocker.patch.object(market_data.yf, "Ticker", return_value=fake_ticker)

    result = WeeklyStockPricesTool().run(
        ticker="DMLRY", start_date="2025-06-01", end_date="2026-06-01"
    )

    assert result.success is False
    assert result.data is None
    assert "DMLRY" in result.error
    assert result.metadata["price_points"] == 0


def test_extract_llm_sources_ignores_failed_calls() -> None:
    result = {
        "messages": [
            _ToolMessage("get_weekly_stock_prices", "Tool 'get_weekly_stock_prices' failed: boom"),
        ]
    }

    assert EconomicsAgent.extract_llm_sources(result) == []
