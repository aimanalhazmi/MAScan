from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from mascan.agents.economics.agent import EconomicsAgent
from mascan.agents.economics.prompts import build_user_prompt
from mascan.agents.economics.tools.market_data import WeeklyStockPricesTool
from mascan.agents.registry import agent_registry
from mascan.contracts.metrics import TokenUsage
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult


def test_economics_agent_is_registered() -> None:
    import mascan.agents.economics  # noqa: F401

    assert "economics" in agent_registry.all_names()


def test_economics_agent_loads_finance_and_web_tools() -> None:
    agent = EconomicsAgent()

    assert "web_search" in agent.optional_tools
    assert "get_weekly_stock_prices" in agent.optional_tools


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
            data=[
                {
                    "title": "EU manufacturing PMI slips in Q4",
                    "url": "https://example.com/eu-pmi",
                    "markdown": "Manufacturing activity contracted.",
                }
            ],
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
        return_value=({}, "Economic outlook findings", []),
    )

    @contextmanager
    def fake_usage_callback(*args: Any, **kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(
            usage_metadata={
                "gpt-test": {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                }
            }
        )

    mocker.patch(
        "mascan.core.metrics.callbacks.get_usage_metadata_callback",
        fake_usage_callback,
    )

    report = agent.run(tasks=["EU manufacturing outlook"])

    assert isinstance(report, AgentReport)
    assert report.agent_name == "economics"
    assert report.tasks == ["EU manufacturing outlook"]
    assert report.findings == "Economic outlook findings"
    assert report.metadata["mode"] == "mixed"
    assert report.metadata["deterministic_tools"] == []
    assert report.component_metrics["economics"].run_count == 1
    assert report.component_metrics["economics"].duration_seconds >= 0
    assert report.component_metrics["economics"].token_usage == TokenUsage(
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
    )
    assert report.component_metrics["economics"].agents["analyst"].token_usage == TokenUsage(
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
    )
    assert "execution" not in report.metadata
    assert "## Economics Analysis" in report.rendered_markdown
    # Sources are real article links labelled by title, not tool names.
    assert [source.url for source in report.sources] == ["https://example.com/eu-pmi"]
    assert report.sources[0].name == "EU manufacturing PMI slips in Q4"
    assert "[EU manufacturing PMI slips in Q4](https://example.com/eu-pmi)" in (
        report.rendered_markdown
    )


class _ToolMessage:
    """Minimal stand-in for a LangChain ToolMessage."""

    def __init__(self, name: str, content: Any) -> None:
        self.type = "tool"
        self.name = name
        self.content = content


def test_extract_llm_sources_surfaces_market_data() -> None:
    # The market-data tool now embeds a `sources` list; extract_llm_sources emits
    # one Source per unique URL from it (deduped across repeated tool calls).
    payload = {
        "ticker": "BMW.DE",
        "start_date": "2025-06-01",
        "end_date": "2026-06-01",
        "fundamentals": {"company_name": "Bayerische Motoren Werke AG"},
        "weekly_prices": [],
        "sources": [
            {
                "name": "Yahoo Finance company summary: BMW.DE",
                "category": "summary",
                "url": "https://finance.yahoo.com/quote/BMW.DE",
            },
            {
                "name": "Yahoo Finance price history: BMW.DE",
                "category": "prices",
                "url": "https://finance.yahoo.com/quote/BMW.DE/history",
            },
        ],
    }
    result = {
        "messages": [
            _ToolMessage("get_weekly_stock_prices", payload),
            _ToolMessage("get_weekly_stock_prices", payload),  # deduped
        ]
    }

    sources = EconomicsAgent.extract_llm_sources(result)

    assert len(sources) == 2
    by_url = {source.url: source for source in sources}
    summary = by_url["https://finance.yahoo.com/quote/BMW.DE"]
    assert summary.name == "Yahoo Finance company summary: BMW.DE"
    assert summary.metadata["ticker"] == "BMW.DE"
    assert summary.metadata["company_name"] == "Bayerische Motoren Werke AG"
    assert summary.metadata["category"] == "summary"

    from mascan.agents.sources import format_source_line

    line = format_source_line(summary)
    assert line == (
        "- [Yahoo Finance company summary: BMW.DE]"
        "(https://finance.yahoo.com/quote/BMW.DE)"
    )


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
