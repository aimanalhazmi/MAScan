import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mascan.contracts.reports import AgentReport, Source
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
3. Cite key factual claims with paper-style numbered citations from the provided
   Citation Registry.
4. If some agents failed, briefly acknowledge gaps but still answer with
   the information you have.
5. Be concise. No filler. No restating the question.

Citation rules:
- Use only citation numbers that appear in the Citation Registry.
- Put a citation after every important factual claim, number, date, regulation,
  policy, market trend, or risk judgment.
- Use exactly this citation format in the body:
  [1](https://source-url)
- Replace "1" and the URL with the correct source number and URL from the
  Citation Registry.
- Do not show raw URLs outside Markdown links in the body.
- Do not invent citation numbers or URLs.
- If a claim lacks source support, state the uncertainty or mark it as
  insufficiently evidenced instead of citing it.
"""

HTML_SOURCE_REF_PATTERN = re.compile(r'href=["\']#source-(\d+)["\']')
MARKDOWN_SOURCE_REF_PATTERN = re.compile(r"\[(\d+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class CitationEntry:
    number: int
    source: Source


@dataclass(frozen=True)
class CitedLink:
    number: int
    url: str


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
    summary = _renumber_citation_links(_normalize_citation_links(state, str(response.content)))
    markdown = _render_markdown(state, summary)

    return {"final_summary": summary, "final_markdown": markdown}


def _build_synthesis_prompt(state: GraphState) -> str:
    parts = [f"User question:\n{state.user_input}\n"]

    citation_registry = _build_citation_registry(state)
    if citation_registry:
        parts.append("Citation Registry:\n")
        for entry in citation_registry:
            parts.append(f"[{entry.number}] {entry.source.name} - {entry.source.url}\n")
        parts.append(
            "\nUse these numbered sources for body citations. For example: "
            "[1](https://source-url)\n"
        )
    else:
        parts.append(
            "Citation Registry:\n"
            "(No URL-backed sources are available. Do not create clickable citations.)\n"
        )

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
            f"  - [{s.name}]({s.url})" if s.url else f"  - {s.name} (no URL)"
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
    """Combine the LLM summary with a paper-style Sources section."""
    parts = [
        "# Final Report\n",
        f"**Query:** {state.user_input}\n",
        "## Summary\n",
        summary,
    ]
    sources_section = _render_sources_section(state, summary)
    if sources_section:
        parts.append("\n## Sources\n")
        parts.append(sources_section)
    return "\n".join(parts)


def _render_sources_section(state: GraphState, summary: str = "") -> str:
    """Render sources cited in the body, falling back to all sources if needed."""
    registry = _build_citation_registry(state)
    if not registry:
        return ""

    by_url = {entry.source.url: entry.source for entry in registry}
    cited_links = _extract_cited_source_links(summary)
    lines: list[str] = []

    if cited_links:
        for cited in cited_links:
            source = by_url.get(cited.url) or Source(name=cited.url, url=cited.url)
            lines.append(_format_numbered_source(cited.number, source))
        return "\n".join(lines)

    return "\n".join(
        _format_numbered_source(entry.number, entry.source) for entry in registry
    )


def _build_citation_registry(state: GraphState) -> list[CitationEntry]:
    """Collect URL-backed sources once, preserving first-seen order."""
    entries: list[CitationEntry] = []
    seen: set[str] = set()
    for report in state.reports.values():
        for source in report.sources:
            if not source.url:
                continue
            key = source.url.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            entries.append(CitationEntry(number=len(entries) + 1, source=source))
    return entries


def _extract_cited_source_numbers(markdown: str) -> list[int]:
    """Return unique source numbers in first-citation order."""
    return [link.number for link in _extract_cited_source_links(markdown)]


def _extract_cited_source_links(markdown: str) -> list[CitedLink]:
    """Return unique Markdown citation links in first-citation order."""
    links: list[CitedLink] = []
    seen_urls: set[str] = set()
    for match in MARKDOWN_SOURCE_REF_PATTERN.finditer(markdown):
        number = int(match.group(1))
        url = match.group(2)
        if url not in seen_urls:
            seen_urls.add(url)
            links.append(CitedLink(number=number, url=url))
    return links


def _extract_html_source_numbers(markdown: str) -> list[int]:
    """Return unique legacy HTML source anchor numbers in first-citation order."""
    numbers: list[int] = []
    seen: set[int] = set()
    for match in HTML_SOURCE_REF_PATTERN.finditer(markdown):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def _format_numbered_source(number: int, source: Source) -> str:
    return f"{number}. [{source.name}]({source.url})"


def _normalize_citation_links(state: GraphState, summary: str) -> str:
    """Convert old HTML in-page citations into OpenWebUI-friendly Markdown links."""
    registry = _build_citation_registry(state)
    urls_by_number = {entry.number: entry.source.url for entry in registry}

    def replace_html_ref(match: re.Match[str]) -> str:
        number = int(match.group(1))
        url = urls_by_number.get(number)
        if not url:
            return match.group(0)
        return f"[{number}]({url})"

    return re.sub(
        r'<sup><a href=["\']#source-(\d+)["\']>\[\d+\]</a></sup>',
        replace_html_ref,
        summary,
    )


def _renumber_citation_links(summary: str) -> str:
    """Renumber Markdown citations by first appearance in the final report body."""
    number_by_url: dict[str, int] = {}

    def replace_markdown_ref(match: re.Match[str]) -> str:
        url = match.group(2)
        if url not in number_by_url:
            number_by_url[url] = len(number_by_url) + 1
        return f"[{number_by_url[url]}]({url})"

    return MARKDOWN_SOURCE_REF_PATTERN.sub(replace_markdown_ref, summary)
