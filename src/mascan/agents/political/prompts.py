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


def render_runtime_context(context: dict[str, Any] | None) -> str:
    runtime = (context or {}).get("runtime")
    if not isinstance(runtime, dict):
        return ""

    # Runtime metadata lets date-relative political analysis stay current.
    return (
        "Runtime context:\n"
        f"- Current date: {runtime.get('current_date')}\n"
        f"- Timezone: {runtime.get('timezone')}\n\n"
    )


def render_retry_feedback(context: dict[str, Any] | None) -> str:
    retry_feedback = (context or {}).get("retry_feedback")
    if not isinstance(retry_feedback, dict):
        return ""

    parts = [
        "Quality gate retry feedback:",
        f"- Status: {retry_feedback.get('status')}",
        f"- Feedback: {retry_feedback.get('feedback')}",
        f"- Retry instruction: {retry_feedback.get('instruction')}",
    ]
    previous_report = retry_feedback.get("previous_report")
    if previous_report:
        parts.append(f"- Previous report to revise:\n{previous_report}")
    return "\n".join(parts) + "\n\n"


def build_user_prompt(
    tasks: list[str],
    tool_block: str,
    context: dict[str, Any] | None = None,
) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    return (
        f"{render_runtime_context(context)}"
        f"{render_retry_feedback(context)}"
        f"Tasks to analyze:\n{task_lines}\n\n"
        f"Information already gathered:\n{tool_block}\n\n"
        "Write a concise political-risk analysis addressing the tasks above. "
        "Cover relevant policy, regulatory, geopolitical, trade, sanction, election, "
        "government-intervention, and industrial-policy factors. Explain business "
        "implications and cite sources by name. Call optional tools only if needed."
    )
