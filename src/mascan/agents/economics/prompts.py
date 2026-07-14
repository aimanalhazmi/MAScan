"""Prompt helpers for the Economics agent.

The system prompt itself lives in config.yaml. This module holds prompt
*construction helpers* — functions that format tool outputs into prompt
sections, build task lists, etc.
"""

from typing import Any

from mascan.agents.context import (
    render_agent_context,
    render_citation_requirements,
    render_runtime_context,
)


def build_user_prompt(
    tasks: list[str],
    tool_block: str,
    context: dict[str, Any] | None = None,
) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    return (
        f"{render_agent_context(context)}"
        f"{render_runtime_context(context)}"
        f"Tasks to analyze:\n{task_lines}\n\n"
        f"Information already gathered:\n{tool_block}\n\n"
        "Write a concise analysis addressing the tasks above. "
        "Use get_weekly_stock_prices when the task mentions a public company, "
        "stock ticker, stock performance, valuation, equity-market impact, or "
        "company-specific market sensitivity. If the user does not provide dates, "
        "use the last 12 months relative to the runtime current date. "
        "Do not use get_weekly_stock_prices for broad sector or macro questions "
        "without a company or ticker.\n\n"
        + render_citation_requirements(
            "For get_weekly_stock_prices, attribute figures to Yahoo Finance and use "
            "the returned ticker URL.",
            "For web_search results, use the page title or publisher and the returned URL.",
            "State the period and metric used for market or price data.",
        )
    )
