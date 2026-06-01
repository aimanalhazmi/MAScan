"""Prompt helpers for the Social Media agent."""

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
        "Write a concise social-media analysis addressing the tasks above. "
        "Group findings by platform when platform evidence differs. Cover sentiment, "
        "recurring user pain points, demand or adoption signals, controversy, and "
        "business implications. Treat Reddit and X as qualitative social evidence, "
        "not statistically representative market research. Distinguish confirmed "
        "public discussion from anecdotal chatter and uncertainty. Cite sources by "
        "name. Call x_search for recent real-time conversation when useful, and call "
        "web_search only when broader public-web context or fallback evidence is needed."
    )
