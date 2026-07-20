"""Prompt helpers for the Legal agent.

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
from mascan.contracts.tools import ToolResult


def render_tool_outputs(outputs: dict[str, ToolResult]) -> str:
    parts: list[str] = []
    for name, result in outputs.items():
        if result.success:
            parts.append(f"### Tool: {name} (source: {result.source})\n{result.data}\n")
        else:
            parts.append(f"### Tool: {name} — FAILED ({result.error})\n")
    return "\n".join(parts)


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
        "Write a concise legal and regulatory analysis addressing the tasks "
        "above. Distinguish enacted law from proposals, note effective dates "
        "and comment deadlines where known, and call optional tools only if needed.\n\n"
        + render_citation_requirements(
            "For Federal Register evidence, cite the returned official document URL.",
            "For EU law, cite the returned EUR-Lex document URL.",
            "For web_search evidence, cite the exact returned page URL.",
            "State whether a cited measure is enacted, proposed, under consultation, "
            "or not yet effective.",
        )
    )
