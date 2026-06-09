"""Prompt helpers for the Economics agent.

The system prompt itself lives in config.yaml. This module holds prompt
*construction helpers* — functions that format tool outputs into prompt
sections, build task lists, etc.
"""

from typing import Any

from mascan.agents.context import render_agent_context, render_runtime_context


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
        "without a company or ticker."
        "\n\nCitation requirements:\n"
        "- Cite evidence directly in the analysis text using Markdown links.\n"
        "- When you use get_weekly_stock_prices, explicitly attribute the figures "
        "to Yahoo Finance and cite the ticker, for example "
        "[Yahoo Finance: BMW.DE](https://finance.yahoo.com/quote/BMW.DE). State the "
        "weekly price/fundamental data you relied on so the market-data tool is "
        "visible in the analysis.\n"
        "- For web_search results, cite the page url with the page title or source name.\n"
        "- If a claim cannot be linked to a source, explicitly mark it as unlinked evidence."
    )
