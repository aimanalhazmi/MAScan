"""Prompt helpers for the Economics agent.

The system prompt itself lives in config.yaml. This module holds prompt
*construction helpers* — functions that format tool outputs into prompt
sections, build task lists, etc.
"""

from mascan.contracts.tools import ToolResult


def render_tool_outputs(outputs: dict[str, ToolResult]) -> str:
    parts: list[str] = []
    for name, result in outputs.items():
        if result.success:
            parts.append(f"### Tool: {name} (source: {result.source})\n{result.data}\n")
        else:
            parts.append(f"### Tool: {name} — FAILED ({result.error})\n")
    return "\n".join(parts)


def build_user_prompt(tasks: list[str], tool_block: str) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    return (
        f"Tasks to analyze:\n{task_lines}\n\n"
        f"Information already gathered:\n{tool_block}\n\n"
        "Write a concise analysis addressing the tasks above. "
        "Cite sources by name. "
        "Use get_weekly_stock_prices when the task mentions a public company, "
        "stock ticker, stock performance, valuation, equity-market impact, or "
        "company-specific market sensitivity. If the user does not provide dates, "
        "use the last 12 months: start_date=2025-05-26 and end_date=2026-05-26. "
        "Do not use get_weekly_stock_prices for broad sector or macro questions "
        "without a company or ticker."
    )
