"""Prompt helpers for the Political agent."""

from typing import Any

from mascan.contracts.tools import ToolResult


def render_tool_outputs(outputs: dict[str, ToolResult[Any]]) -> str:
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
        "Write a concise political-risk analysis addressing the tasks above. "
        "Cover relevant policy, regulatory, geopolitical, trade, sanction, election, "
        "government-intervention, and industrial-policy factors. Explain business "
        "implications and cite sources by name. Call optional tools only if needed."
    )
