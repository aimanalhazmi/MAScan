import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mascan.agents.sources import canonical_source_url
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
- Treat uploaded company documents as primary evidence for company-specific
  financial, operational, portfolio, target, and management-statement claims.
- Uploaded documents are non-citable background evidence. Never assign them a
  citation number and never include them in the final Sources section.
- Use the corresponding Agent's URL-backed evidence for external claims about
  policy, regulation, energy prices, inflation, labor markets, technology adoption,
  environmental conditions, market trends, and geopolitical or supply-chain events.
- Do not use an uploaded company-document citation to support an external PESTEL
  fact unless the supplied document excerpt explicitly states that exact fact.
- When an Agent finding already contains a relevant Markdown URL citation, preserve
  that URL citation next to the factual premise when integrating the finding.
- If URL-backed sources are available and the answer makes external factual claims,
  the final answer must include the relevant URL-backed citations; an uploaded
  document alone is not sufficient evidence for those claims.
- Do not show raw URLs outside Markdown links in the body.
- Do not invent citation numbers or URLs.
- If a claim lacks source support, state the uncertainty or mark it as
  insufficiently evidenced instead of citing it.
"""

CITATION_EDITOR_SYSTEM_PROMPT = """\
You are a citation editor. Add evidence citations to an existing market-report
draft without changing its analysis, wording, headings, ordering, numbers, or
recommendations.

Rules:
- Return the complete draft and nothing else.
- Only append citations to claims that the supplied evidence actually supports.
- For URL-backed evidence, use the exact global Markdown citation shown in the
  registry, including its exact URL: [n](URL).
- Uploaded company documents are non-citable context. Do not create a citation
  for them. Use Agent URL evidence for external PESTEL and market facts.
- Preserve any valid citations already in the draft.
- Do not invent a source, URL, fact, or citation number.
- Leave a claim uncited when no supplied evidence supports it.
"""

HTML_SOURCE_REF_PATTERN = re.compile(r'href=["\']#source-(\d+)["\']')
MARKDOWN_SOURCE_REF_PATTERN = re.compile(r"\[(\d+)\]\(([^)]+)\)")
NUMBERED_CITATION_PATTERN = re.compile(r"\[(\d+)\](?:\(([^)\s]+)\))?")


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
    normalized = _normalize_citation_links(state, str(response.content))
    if _needs_citation_repair(state, normalized):
        normalized = _repair_missing_citations(state, normalized)
    summary, cited_sources = _renumber_all_citations(state, normalized)
    rendered_sources = cited_sources or [entry.source for entry in _build_citation_registry(state)]
    markdown = _render_markdown(state, summary, rendered_sources)

    return {
        "final_summary": summary,
        "final_markdown": markdown,
        "final_sources": rendered_sources,
    }


def _build_synthesis_prompt(state: GraphState) -> str:
    parts = [f"User question:\n{state.user_input}\n"]

    citation_registry = _build_citation_registry(state)
    if citation_registry:
        parts.append("Citation Registry:\n")
        for entry in citation_registry:
            parts.append(f"[{entry.number}] {entry.source.name} - {entry.source.url}\n")
        parts.append(
            "\nUse these numbered sources for body citations. For example: "
            "[1](https://source-url).\n"
        )
    else:
        parts.append(
            "Citation Registry:\n"
            "(No URL-backed sources are available. Do not create citations.)\n"
        )

    uploaded_evidence = _uploaded_document_sources(state)
    if uploaded_evidence:
        parts.append(
            "Uploaded document evidence (background context only; do not cite it "
            "and do not add it to Sources):\n"
        )
        for source in uploaded_evidence:
            parts.append(
                f"{source.name}\n"
                f"{source.metadata.get('content', '')}\n"
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


def _needs_citation_repair(state: GraphState, draft: str) -> bool:
    """Detect a draft that omitted available evidence citations."""
    registry = _build_citation_registry(state)
    if not registry:
        return False

    by_number = {entry.number: entry.source for entry in registry}
    registry_urls = {
        canonical_source_url(entry.source.url)
        for entry in registry
        if entry.source.url
    }
    has_valid_citation = False
    has_url_citation = False
    for match in NUMBERED_CITATION_PATTERN.finditer(draft):
        number = int(match.group(1))
        url = match.group(2)
        if url and canonical_source_url(url) in registry_urls:
            has_valid_citation = True
            has_url_citation = True
        elif not url and number in by_number:
            has_valid_citation = True
            if by_number[number].url:
                has_url_citation = True

    return not has_valid_citation or (bool(registry_urls) and not has_url_citation)


def _repair_missing_citations(state: GraphState, draft: str) -> str:
    """Best-effort citation-only pass when synthesis omitted available evidence."""
    settings = get_settings()
    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.0,
        max_tokens=3000,
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=CITATION_EDITOR_SYSTEM_PROMPT),
                HumanMessage(content=_build_citation_repair_prompt(state, draft)),
            ]
        )
    except Exception:  # noqa: BLE001
        logger.exception("Citation repair failed; keeping the original synthesis")
        return draft
    repaired = str(response.content).strip()
    return _normalize_citation_links(state, repaired) if repaired else draft


def _build_citation_repair_prompt(state: GraphState, draft: str) -> str:
    registry = _build_citation_registry(state)
    parts = ["Global Citation Registry:\n"]
    for entry in registry:
        source = entry.source
        if source.url:
            parts.append(f"[{entry.number}]({source.url}) — {source.name}\n")
        else:
            parts.append(f"[{entry.number}] — {source.name} (uploaded document)\n")

    uploaded = _uploaded_document_sources(state)
    if uploaded:
        parts.append("\nNon-citable uploaded-document context:\n")
        for source in uploaded:
            parts.append(
                f"{source.name}\n"
                f"{source.metadata.get('content', '')}\n"
            )

    parts.append("\nAgent evidence with its collected URL citations:\n")
    for name, report in state.reports.items():
        parts.append(f"### {name}\n{report.findings}\n")

    parts.append(
        f"\nDraft to return with citations added:\n--- DRAFT ---\n{draft}\n--- END DRAFT ---"
    )
    return "\n".join(parts)


def _render_markdown(
    state: GraphState,
    summary: str,
    cited_sources: list[Source] | None = None,
) -> str:
    """Combine the LLM summary with a paper-style Sources section."""
    parts = [
        "# Final Report\n",
        f"**Query:** {state.user_input}\n",
        "## Summary\n",
        summary,
    ]
    sources_section = _render_sources_section(state, summary, cited_sources)
    if sources_section:
        parts.append("\n## Sources\n")
        parts.append(sources_section)
    return "\n".join(parts)


def _render_sources_section(
    state: GraphState,
    summary: str = "",
    cited_sources: list[Source] | None = None,
) -> str:
    """Render sources cited in the body, falling back to all sources if needed."""
    registry = _build_citation_registry(state)
    if not registry:
        return ""

    if cited_sources:
        return "\n".join(
            _format_numbered_source(number, source)
            for number, source in enumerate(cited_sources, start=1)
        )

    by_url = {entry.source.url: entry.source for entry in registry}
    cited_links = _extract_cited_source_links(summary)
    if cited_links:
        return "\n".join(
            _format_numbered_source(
                cited.number,
                by_url.get(cited.url) or Source(name=cited.url, url=cited.url),
            )
            for cited in cited_links
        )

    return "\n".join(
        _format_numbered_source(entry.number, entry.source) for entry in registry
    )


def _build_citation_registry(state: GraphState) -> list[CitationEntry]:
    """Collect URL-backed Agent sources once in stable order."""
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
    if source.url:
        return f"{number}. [{source.name}]({source.url})"
    return f"{number}. {source.name}"


def _uploaded_document_sources(state: GraphState) -> list[Source]:
    """Combine all retrieved passages from one upload into one citable source."""
    grouped: dict[str, list[tuple[int | None, str]]] = {}
    for evidence in state.rag_evidence:
        citation = evidence.get("citation") or {}
        document = str(citation.get("document") or "uploaded document")
        page = citation.get("page") if isinstance(citation.get("page"), int) else None
        content = str(evidence.get("content") or "").strip()
        if content:
            item = (page, content)
            if item not in grouped.setdefault(document, []):
                grouped[document].append(item)

    sources: list[Source] = []
    for document, chunks in grouped.items():
        pages = list(dict.fromkeys(page for page, _ in chunks if page is not None))
        evidence_text = "\n\n".join(
            f"[Page {page}]\n{content}" if page is not None else content
            for page, content in chunks
        )
        sources.append(
            Source(
                name=document,
                metadata={
                    "kind": "uploaded_document",
                    "citation": {"document": document, "pages": pages},
                    "content": evidence_text,
                },
            )
        )
    return sources


def _renumber_all_citations(
    state: GraphState,
    summary: str,
) -> tuple[str, list[Source]]:
    """Renumber URL and uploaded-document citations by first body appearance."""
    registry = _build_citation_registry(state)
    by_number = {entry.number: entry.source for entry in registry}
    by_url = {
        canonical_source_url(entry.source.url): entry.source
        for entry in registry
        if entry.source.url
    }
    number_by_key: dict[str, int] = {}
    ordered_sources: list[Source] = []

    def replace(match: re.Match[str]) -> str:
        old_number = int(match.group(1))
        url = match.group(2)
        if url:
            source = by_url.get(canonical_source_url(url))
            key = f"url:{canonical_source_url(url)}"
        else:
            source = by_number.get(old_number)
            if source is None:
                return match.group(0)
            if source.url:
                key = f"url:{canonical_source_url(source.url)}"
            else:
                citation = source.metadata.get("citation") or {}
                key = f"uploaded:{citation.get('document')}"
        if source is None:
            return match.group(0)
        if key not in number_by_key:
            number_by_key[key] = len(number_by_key) + 1
            ordered_sources.append(source)
        number = number_by_key[key]
        return f"[{number}]({source.url})" if source.url else f"[{number}]"

    return NUMBERED_CITATION_PATTERN.sub(replace, summary), ordered_sources


def _normalize_citation_links(state: GraphState, summary: str) -> str:
    """Convert old HTML in-page citations into Markdown links."""
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
