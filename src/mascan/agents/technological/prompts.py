"""Prompt helpers for the Technological agent."""

from typing import Any

from mascan.agents.context import (
    render_agent_context,
    render_citation_requirements,
    render_runtime_context,
)


def build_user_prompt(
    tasks: list[str],
    tool_block: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    task_lines = "\n".join(f"- {task}" for task in tasks)
    info = f"Information already gathered:\n{tool_block}\n\n" if tool_block else ""
    return (
        f"{render_agent_context(context)}"
        f"{render_runtime_context(context)}"
        f"Tasks to analyze:\n{task_lines}\n\n"
        f"{info}"
        "For each task, assess the relevant technological dimensions, including R&D, "
        "patents, scientific research, digital infrastructure, AI and automation, "
        "cybersecurity, technical talent, startup ecosystems, and technology standards. "
        "Use scholar_search for relevant research evidence and do not call it more than "
        "three times. Use web_search for current adoption, infrastructure, commercial, "
        "industry, and ecosystem evidence. Report objective evidence before interpretation. "
        "Distinguish established technologies from emerging or experimental technologies, "
        "and distinguish observed measurements from forecasts and industry estimates. "
        "Explain the direct business implications for competitiveness, operations, "
        "investment, product development, and market entry.\n\n"
        + render_citation_requirements(
            "For scholar_search, cite the exact returned paper or publication URL.",
            "For web_search, cite the exact returned page URL.",
            "State the coverage period, geography, metric, or methodology where relevant.",
        )
    )
