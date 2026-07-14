"""Shared prompt context renderers for domain agents."""

from typing import Any

CITATION_REQUIREMENTS = """\
Citation requirements:
- Before writing the final analysis, if the tasks require external verifiable facts
  and the supplied context contains no URL-backed evidence for them, call at least
  one suitable evidence tool (web_search or a domain-specific official-source tool).
- Make no more than two targeted evidence calls for this purpose, then write the
  report; do not keep searching recursively.
- Cite every important factual claim, number, date, regulation, policy,
  market trend, research result, or evidence-based risk judgment directly
  in the analysis text.
- Use a Markdown link immediately after the supported statement:
  [Source name](exact URL returned by the tool).
- Use only URLs returned by tools during this agent run.
- Do not invent, reconstruct, shorten, or guess a URL.
- Reuse the same URL when multiple statements rely on the same source.
- Do not assign citation numbers and do not create a Sources section;
  the application will number, deduplicate, and render Sources.
"""


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


def render_citation_requirements(*agent_rules: str) -> str:
    """Return the shared citation contract plus domain-specific bullet rules."""
    rules = "\n".join(f"- {rule}" for rule in agent_rules if rule)
    return CITATION_REQUIREMENTS + (rules + "\n" if rules else "")
