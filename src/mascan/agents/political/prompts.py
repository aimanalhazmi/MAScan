"""Prompt helpers for the Political agent."""

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
        "Write a concise political-risk analysis addressing the tasks above. "
        "Cover relevant policy, regulatory, geopolitical, trade, sanction, election, "
        "government-intervention, and industrial-policy factors. Explain business "
        "implications and call optional tools only if needed.\n\n"
        + render_citation_requirements(
            "For web_search and news_api evidence, use the article title or publisher "
            "as the Markdown link label and use the returned article URL.",
            "Distinguish enacted policies and confirmed government actions from "
            "proposals, campaign positions, or speculation.",
        )
    )
