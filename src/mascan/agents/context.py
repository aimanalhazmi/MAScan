"""Shared prompt context renderers for domain agents."""

from typing import Any


def render_tool_outputs(outputs: dict[str, Any]) -> str:
    parts: list[str] = []
    for name, result in outputs.items():
        if result.success:
            parts.append(f"### Tool: {name} (source: {result.source})\n{result.data}\n")
        else:
            parts.append(f"### Tool: {name} — FAILED ({result.error})\n")
    return "\n".join(parts)


def render_runtime_context(context: dict[str, Any] | None) -> str:
    runtime = (context or {}).get("runtime")
    if not isinstance(runtime, dict):
        return ""

    return (
        "Runtime context:\n"
        f"- Current date: {runtime.get('current_date')}\n"
        f"- Timezone: {runtime.get('timezone')}\n\n"
    )


def render_agent_context(context: dict[str, Any] | None) -> str:
    values = context or {}
    parts: list[str] = []
    objective_context = values.get("objective_context")
    if isinstance(objective_context, str) and objective_context.strip():
        parts.append(f"Domain objective:\n{objective_context.strip()}\n")

    return "\n\n".join(parts) + ("\n\n" if parts else "")
