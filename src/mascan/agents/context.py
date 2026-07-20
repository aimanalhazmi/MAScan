"""Shared prompt context renderers for domain agents."""

from typing import Any

from mascan.contracts.reports import Source

CITATION_REQUIREMENTS = """\
Citation requirements:
- Before writing the final analysis, if the tasks require external verifiable facts
  and the supplied context contains no evidence for them, call at least
  one suitable evidence tool (web_search or a domain-specific official-source tool).
- Make no more than two targeted evidence calls for this purpose, then write the
  report; do not keep searching recursively.
- Cite every important factual claim, number, date, regulation, policy,
  market trend, research result, or evidence-based risk judgment directly
  in the analysis text.
- Use a Markdown link immediately after the supported statement:
  [Source name](exact provided URL).
- Use only exact URLs returned by tools during this agent run or exact uploaded-file
  links shown in the supplied context.
- An uploaded document supports only facts stated in its supplied excerpts.
- If the user explicitly requires an uploaded document and its evidence is relevant
  to your assigned tasks, use and cite that evidence in the report.
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
    user_input = values.get("user_input")
    if isinstance(user_input, str) and user_input.strip():
        parts.append(f"Original user request:\n{user_input.strip()}\n")

    objective_context = values.get("objective_context")
    if isinstance(objective_context, str) and objective_context.strip():
        parts.append(f"Domain objective:\n{objective_context.strip()}\n")

    provided_sources = values.get("provided_sources")
    if isinstance(provided_sources, list):
        rendered_sources: list[str] = []
        for value in provided_sources:
            try:
                source = value if isinstance(value, Source) else Source.model_validate(value)
            except (TypeError, ValueError):
                continue
            if not source.url:
                continue
            content = str(source.metadata.get("content") or "").strip()
            if not content:
                continue
            rendered_sources.append(
                f"### [{source.name}]({source.url})\n{content}"
            )
        if rendered_sources:
            parts.append(
                "Uploaded document evidence supplied by the planner. Use only excerpts "
                "relevant to your tasks and cite the exact file link shown:\n"
                + "\n\n".join(rendered_sources)
                + "\n"
            )

    return "\n\n".join(parts) + ("\n\n" if parts else "")


def render_citation_requirements(*agent_rules: str) -> str:
    """Return the shared citation contract plus domain-specific bullet rules."""
    rules = "\n".join(f"- {rule}" for rule in agent_rules if rule)
    return CITATION_REQUIREMENTS + (rules + "\n" if rules else "")
