"""Prompt helpers for the Social agent."""

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
        "Write a concise PESTEL social analysis addressing the tasks above. "
        "Use official indicators for demographic, education, health, labour, "
        "inequality, and social baseline context. Use Reddit and X as qualitative "
        "public-discussion signals, not statistically representative market research. "
        "Group findings by evidence type when useful: official social indicators, "
        "public discourse, user pain points, demand/adoption signals, controversy, "
        "and business implications. Distinguish official statistics from anecdotal "
        "chatter and uncertainty. "
        "\n\nCitation requirements:\n"
        "- Cite evidence directly in the analysis text using Markdown links.\n"
        "- For World Bank indicators, cite the attached api_url, for example "
        "[World Bank: unemployment](https://api.worldbank.org/.../SL.UEM.TOTL.ZS).\n"
        "- For web_search results, cite the page url with the page title or source name.\n"
        "- For Reddit and X results, cite each referenced post with its url when available.\n"
        "- Do not write bare citations like '(World Bank, 2024)', '(source: Reddit)', "
        "or '(source: X)' when a URL is present in the tool output.\n"
        "- If a claim cannot be linked to a URL, explicitly mark it as unlinked evidence.\n\n"
        "Use the already gathered web_search and World Bank evidence as the baseline. "
        "When qualitative community or recent-post signals would strengthen the analysis, "
        "call the reddit_search and/or x_search tools yourself; otherwise rely on the "
        "baseline evidence. Only cite Reddit or X posts that those tool calls actually return."
    )
