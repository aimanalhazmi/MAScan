"""Prompt helpers for the Social agent."""

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
        "Write a concise PESTEL social analysis addressing the tasks above. "
        "When the case is place-based, rural, agricultural, wine, tourism, or "
        "regional-development related, explicitly address: (1) jobs/employment, "
        "(2) tourism or visitor economy if relevant, (3) education/skills, and "
        "(4) rural community change or depopulation — with business impact. "
        "Do not replace those themes with only urbanization, health-consciousness, "
        "or generic eco-consumer awareness filler. "
        "Use official indicators for demographic, education, health, labour/jobs, "
        "inequality, tourism where relevant, rural/community change, and social "
        "baseline context. Use Reddit and X as qualitative "
        "public-discussion signals, not statistically representative market research. "
        "Group findings by evidence type when useful: official social indicators, "
        "public discourse, user pain points, demand/adoption signals, controversy, "
        "and business implications. Distinguish official statistics from anecdotal "
        "chatter and uncertainty.\n\n"
        "Use the already gathered web_search and World Bank evidence as the baseline. "
        "When qualitative community or recent-post signals would strengthen the analysis, "
        "call the reddit_search and/or x_search tools yourself; otherwise rely on the "
        "baseline evidence. Only cite Reddit or X posts that those tool calls actually return.\n\n"
        + render_citation_requirements(
            "For World Bank indicators, cite the returned user-facing url, not api_url.",
            "For Reddit and X evidence, cite the exact returned post URL.",
            "Do not use bare citations such as '(World Bank, 2024)' when a URL exists.",
            "Treat Reddit and X as qualitative, non-representative evidence.",
        )
    )
