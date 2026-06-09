"""Prompt helpers for the Political agent."""

from typing import Any

from mascan.contracts.tools import ToolResult
from mascan.agents.context import render_agent_context, render_runtime_context


def build_user_prompt(
    tasks: list[str],
    tool_block: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    info = ""
    if tool_block:
        info = f"Information already gathered:\n{tool_block}\n\n"
    return (
        f"{render_agent_context(context)}"
        f"{render_runtime_context(context)}"
        f"Tasks to analyze:\n{task_lines}\n\n"
        f"{info}"
        "For each task, assess which environmental dimensions are relevant "
        "(climate, emissions, resources, extreme weather, air/water quality, "
        "biodiversity) and call the appropriate tools. "
        "Use get_climate_weather when the task involves temperature, precipitation, "
        "seasonal patterns, or extreme weather events for a specific location or region. "
        "Use get_air_quality when the task involves pollution levels, air quality indices, "
        "or atmospheric conditions affecting operations or workforce. "
        "Use get_resource_availability when the task involves water stress, freshwater "
        "access, forest coverage, CO₂ emissions, or natural resource constraints by country. "
        "Use web_search for biodiversity, land use change, or ecological footprint data "
        "where no dedicated tool is available. "
        "Report raw data findings first, then interpret their business implications. "
        "Cite all sources by name, coverage period, and geographic scope."
    )
