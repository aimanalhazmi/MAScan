from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mascan.contracts.reports import AgentReport
from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger
from mascan.core.settings import get_settings
from mascan.orchestrator.state import GraphState

logger = get_logger("orchestrator.synthesizer")

SYNTHESIZER_SYSTEM_PROMPT = """\
You are the synthesizer of a PESTEL multi-agent market-analysis system.
You receive findings from one or more specialist agents. Your job:

1. Write a coherent, well-structured final answer to the user's question.
2. Integrate insights across dimensions — do not just concatenate.
3. Cite which agent contributed each major point (e.g. "Economics finds...").
4. If some agents failed, briefly acknowledge gaps but still answer with
   the information you have.
5. Be concise. No filler. No restating the question.
"""


def synthesizer_node(state: GraphState) -> dict[str, Any]:
    """LangGraph node: produce the final summary and markdown."""
    if not state.reports and not state.failures:
        logger.warning("Synthesizer ran with no reports and no failures.")
        return {
            "final_summary": "(no agents produced output)",
            "final_markdown": "## Final Report\n\n_No agents produced output._\n",
        }

    settings = get_settings()
    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.3,
        max_tokens=2500,
    )

    user_prompt = _build_synthesis_prompt(state)
    response = llm.invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    summary = str(response.content)
    markdown = _render_markdown(state, summary)

    return {"final_summary": summary, "final_markdown": markdown}


def _build_synthesis_prompt(state: GraphState) -> str:
    parts = [f"User question:\n{state.user_input}\n"]

    if state.reports:
        parts.append("Agent findings:\n")
        for name, report in state.reports.items():
            parts.append(_format_report_for_prompt(name, report))

    if state.failures:
        parts.append("Agents that failed (please note gaps in your answer):\n")
        for name, err in state.failures.items():
            parts.append(f"- {name}: {err}\n")

    return "\n".join(parts)


def _format_report_for_prompt(name: str, report: AgentReport) -> str:
    src_block = (
        "\n".join(
            f"  - {s.name}: {s.url}" if s.url else f"  - {s.name}"
            for s in report.sources
        )
        or "  (no sources)"
    )
    return (
        f"### Agent: {name} (confidence={report.confidence:.2f})\n"
        f"{report.findings}\n"
        f"Sources:\n{src_block}\n"
    )


def _render_markdown(state: GraphState, summary: str) -> str:
    """Combine the LLM summary with an aggregated Sources section."""
    parts = [
        "# Final Report\n",
        f"**Query:** {state.user_input}\n",
        "## Summary\n",
        summary,
    ]
    sources_section = _render_sources_section(state)
    if sources_section:
        parts.append("\n## Sources\n")
        parts.append(sources_section)
    return "\n".join(parts)


def _render_sources_section(state: GraphState) -> str:
    """Gather every agent's real reference links into one grouped, deduped list."""
    blocks: list[str] = []
    for name, report in state.reports.items():
        seen: set[str] = set()
        lines: list[str] = []
        for source in report.sources:
            key = source.url or source.name
            if key in seen:
                continue
            seen.add(key)
            if source.url:
                lines.append(f"- [{source.name}]({source.url})")
            else:
                lines.append(f"- {source.name}")
        if lines:
            blocks.append(f"**{name.title()}**\n" + "\n".join(lines))
    return "\n\n".join(blocks)
