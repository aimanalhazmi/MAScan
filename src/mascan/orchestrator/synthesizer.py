import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mascan.agents.sources import canonical_source_url
from mascan.contracts.reports import AgentReport, Source
from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger
from mascan.core.metrics import measure_component
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
5. Cut filler and do not restate the question — but never drop an evidenced
   factor for the sake of brevity. Completeness of evidenced factors comes first.

Structure and analytical depth:
- When the user asks for a PESTEL analysis (or already lists PESTEL headings),
  use exactly these Markdown headings in order:
  Political, Economic, Social, Technological, Environmental, Legal,
  Strategic implications.
- Under each PESTEL heading, write concise bullets that connect each external
  factor to direct company or business impact — not a bare list of factors.
- Under Strategic implications, give concrete risks, opportunities, or actions
  grounded in the preceding PESTEL sections.
- Give every evidenced factor a concrete business linkage. Do not trade factor
  coverage for depth or depth for coverage — do both: include each evidenced
  factor AND connect it to a business impact.

Theme coverage (when writing a PESTEL report):
- Cover each heading with every concrete factor the agents evidenced for it, not
  just one. Do not leave a heading as only generic wording.
- If the prompt lists Coverage checklist factors, each factor supported by agent
  findings must appear under the heading it genuinely belongs to; do not omit it
  for brevity. The checklist is not the full set of factors to report — cover any
  other evidenced factor the agents found as well.
- Political: policy, government pressure, subsidies, geopolitics, or political risk.
- Economic: prices, demand, costs, incentives, efficiency or operating-cost savings.
- Social: stakeholders, labor/jobs, community, demographics, education, or tourism
  where relevant to the case.
- Technological: process/product innovation, digital systems, infrastructure, or
  R&D adoption with business impact.
- Environmental: emissions, waste, water/resource stress, climate, or pollution
  constraints on operations.
- Legal: specific regulations, permits, fines, compliance duties, or litigation risk
  — not only a vague "compliance" line.
- Keep each factor under its correct PESTEL heading (do not move legal fines into
  Economic, or social labor themes into Political, etc.).
- If agent evidence for a theme is thin, state the gap briefly under that heading
  rather than inventing unsupported detail.

Evidence discipline (reduce unsupported claims):
- Every important factual bullet must be grounded in agent findings or the Citation
  Registry. Include every factor the agents evidenced; cut speculative commentary,
  never evidenced substance. Breadth is only limited by what the agents supported.
- Do not invent regional political tensions, consumer-health trends, product-mix
  shifts, urbanization statistics, or automation benefits unless agent evidence
  states them for this case.
- For Social, prioritize jobs/employment, tourism (when place-based), education,
  and rural community/depopulation impacts when relevant — do not substitute only
  generic "educated urban consumers" filler.
- Strategic implications must reuse risks/opportunities already supported above;
  do not add new unsupported actions.

Citation rules:
- Preserve relevant inline citations already present in Agent findings.
- Cite important factual claims using only the exact numbered Markdown links in
  the Citation Registry, including `/rag/files/...` links.
- Put each citation immediately after the claim it supports.
- Do not invent or alter citation numbers, URLs, sources, or facts.
- If no Registry source supports a claim, state the uncertainty instead of citing it.
"""

CITATION_EDITOR_SYSTEM_PROMPT = """\
You are a citation editor. Add evidence citations to an existing market-report
draft without changing its analysis, wording, headings, ordering, numbers, or
recommendations.

Rules:
- Return the complete draft and nothing else; do not repeat any DRAFT wrapper.
- Only append exact Registry citations to existing claims that the supplied Agent
  evidence actually supports.
- Preserve any valid citations already in the draft.
- Do not add or rewrite facts, analysis, headings, recommendations, sources, URLs,
  or citation numbers.
- Leave a claim uncited when no supplied evidence supports it.
"""

COVERAGE_REPAIR_SYSTEM_PROMPT = """\
You are a PESTEL filing editor. Your job is to aggressively REFILE existing
evidence into the correct headings without inventing new content.

You receive a draft report plus agent evidence. Do this:

1. MOVE / REFILE first (highest priority):
   - Scan the draft AND agent findings for claims already present.
   - Place each claim under the correct heading:
     - Legal: fines, penalties, permits, authorizations, named regulations/directives
     - Economic: prices, demand, costs, incentives, efficiency/operating-cost savings
     - Social: jobs/employment, tourism, education, community, demographics, rural change
     - Political: policy, government pressure, subsidies, geopolitics
     - Technological: named tech, digital systems, infrastructure, R&D adoption
     - Environmental: emissions, waste, water/resource stress, climate, pollution
   - If a claim appears under the wrong heading, move it (do not leave a duplicate).
   - If agent findings already contain a theme that is missing from the correct
     heading, paraphrase that existing agent text into one bullet under the right
     heading. Do not generalize beyond what the agent said.
   - Use the Coverage checklist (when present) as an additional, non-exhaustive set
     of factors to verify: for each checklist factor the agent findings support but
     the draft omits, paraphrase the supporting agent text into one bullet under the
     heading that factor genuinely belongs to (not the heading of the agent that
     raised it). Never add a checklist factor that no agent finding supports, and
     never drop an evidenced factor just because it is absent from the checklist.

2. REMOVE speculative unsupported bullets (for example invented regional tensions,
   generic health/urbanization stats, product-mix trends, or automation benefits
   with no agent support). Prefer dropping them over rewriting into new claims.

3. Do NOT invent facts, numbers, regulations, citations, or new strategic actions.
   Do NOT add filler to "complete" a heading when agent evidence lacks that theme.
   A short evidence-gap note is allowed only when the heading would otherwise be empty.

4. Keep heading order:
   Political, Economic, Social, Technological, Environmental, Legal,
   Strategic implications.
   Strategic implications may only restate supported risks/opportunities already
   present after refiling.

5. Preserve valid citations. When moving/paraphrasing an evidenced claim, keep or
   attach [n](URL) from the Citation Registry when a matching source exists.

6. Return the complete revised draft and nothing else.
"""

PESTEL_HEADINGS: tuple[str, ...] = (
    "Political",
    "Economic",
    "Social",
    "Technological",
    "Environmental",
    "Legal",
)

# Heuristic theme cues used only to decide whether coverage repair should run.
_HEADING_THEME_CUES: dict[str, tuple[str, ...]] = {
    "Political": ("policy", "government", "subsid", "geopolitic", "political"),
    "Economic": ("price", "demand", "cost", "incentive", "efficien", "saving"),
    "Social": ("job", "labor", "labour", "tourism", "education", "community", "demographic"),
    "Technological": (
        "innov",
        "digital",
        "infrastruct",
        "r&d",
        "technolog",
        "battery",
        "microgrid",
    ),
    "Environmental": ("emission", "waste", "water", "climate", "pollut", "aquifer", "co2"),
    "Legal": (
        "regulation",
        "directive",
        "permit",
        "authoriz",
        "fine",
        "penalt",
        "litigation",
        "law",
    ),
}

HTML_SOURCE_REF_PATTERN = re.compile(r'href=["\']#source-(\d+)["\']')
MARKDOWN_SOURCE_REF_PATTERN = re.compile(r"\[(\d+)\]\(([^)]+)\)")
NUMBERED_CITATION_PATTERN = re.compile(r"\[(\d+)\](?:\(([^)\s]+)\))?")
_HEADING_SPLIT_PATTERN = re.compile(
    r"(?im)^(?:#{2,3}\s*)?(Political|Economic|Social|Technological|Environmental|Legal|"
    r"Strategic implications)\s*$"
)


@dataclass(frozen=True)
class CitationEntry:
    number: int
    source: Source


@dataclass(frozen=True)
class CitedLink:
    number: int
    url: str


def synthesizer_node(state: GraphState) -> dict[str, Any]:
    """Measure one synthesizer execution and return its state update."""
    return measure_component("synthesizer", lambda: _synthesizer_node(state))


def _synthesizer_node(state: GraphState) -> dict[str, Any]:
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
        max_tokens=4000,
    )

    user_prompt = _build_synthesis_prompt(state)
    response = llm.invoke(
        [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    normalized = _normalize_citation_links(state, str(response.content))
    if _needs_citation_repair(state, normalized):
        normalized = _repair_missing_citations(state, normalized)
    summary, cited_sources = _renumber_all_citations(state, normalized)
    rendered_sources = cited_sources
    markdown = _render_markdown(state, summary, rendered_sources)

    return {
        "final_summary": summary,
        "final_markdown": markdown,
        "final_sources": rendered_sources,
    }


def _build_coverage_checklist(state: GraphState) -> str:
    """Render planning-time salient factors as an additive synthesis checklist.

    The list is deliberately unlabelled: which agent raised a factor says nothing
    about which PESTEL heading it belongs under (a policy agent routinely surfaces
    fines, which belong under Legal), so labelling factors by their owning agent
    would force miscategorisation.
    """
    factors: list[str] = []
    seen: set[str] = set()
    for assignment in state.plan.values():
        for factor in assignment.salient_factors:
            cleaned = factor.strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                factors.append(cleaned)

    if not factors:
        return ""

    factor_lines = "\n".join(f"- {factor}" for factor in factors)
    return (
        "Coverage checklist (candidate factors identified during planning):\n"
        "Treat this as a floor, not a ceiling. Include every factor below that the "
        "agent findings support, AND any other material factor the agents found that "
        "is not listed. File each factor under whichever PESTEL heading it genuinely "
        "belongs to, judged from the factor itself and never from which agent raised "
        "it — fines, permits, and named regulations belong under Legal even when a "
        "policy agent surfaced them. Never include a factor that has no support in "
        "the agent findings.\n"
        f"{factor_lines}\n"
    )


def _build_synthesis_prompt(state: GraphState) -> str:
    parts = [f"User question:\n{state.user_input}\n"]

    coverage_checklist = _build_coverage_checklist(state)
    if coverage_checklist:
        parts.append(coverage_checklist)

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
            "Citation Registry:\n(No URL-backed sources are available. Do not create citations.)\n"
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
        canonical_source_url(entry.source.url) for entry in registry if entry.source.url
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
        max_tokens=4000,
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
    repaired = _strip_draft_wrappers(str(response.content))
    return _normalize_citation_links(state, repaired) if repaired else draft


def _strip_draft_wrappers(content: str) -> str:
    """Remove exact repair-prompt delimiters if the model echoes them."""
    lines = content.strip().splitlines()
    if lines and lines[0].strip().casefold() == "--- draft ---":
        lines.pop(0)
    if lines and lines[-1].strip().casefold() == "--- end draft ---":
        lines.pop()
    return "\n".join(lines).strip()


def _needs_coverage_repair(draft: str) -> bool:
    """Return True for PESTEL drafts that should get a refiling pass."""
    sections = _split_pestel_sections(draft)
    pestel_present = sum(1 for heading in PESTEL_HEADINGS if heading in sections)
    # Full(ish) PESTEL drafts always get a move/file pass so misplaced claims
    # are corrected even when each heading already looks "theme-full".
    if pestel_present >= 5:
        return True
    if pestel_present < 4:
        return False
    for heading in PESTEL_HEADINGS:
        body = sections.get(heading, "")
        if not body.strip():
            return True
        cues = _HEADING_THEME_CUES[heading]
        lowered = body.lower()
        if not any(cue in lowered for cue in cues):
            return True
    return False


def _split_pestel_sections(draft: str) -> dict[str, str]:
    """Split a draft into heading -> body text for PESTEL sections."""
    matches = list(_HEADING_SPLIT_PATTERN.finditer(draft))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(draft)
        sections[heading] = draft[start:end]
    return sections


def _repair_theme_coverage(state: GraphState, draft: str) -> str:
    """Best-effort pass to fix misplaced/missing PESTEL themes from agent evidence."""
    settings = get_settings()
    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.0,
        max_tokens=4000,
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=COVERAGE_REPAIR_SYSTEM_PROMPT),
                HumanMessage(content=_build_coverage_repair_prompt(state, draft)),
            ]
        )
    except Exception:  # noqa: BLE001
        logger.exception("Coverage repair failed; keeping the current synthesis")
        return draft
    repaired = str(response.content).strip()
    return repaired or draft


def _build_coverage_repair_prompt(state: GraphState, draft: str) -> str:
    parts = [
        "Repair PESTEL theme coverage and category placement using only the evidence below.\n",
    ]
    parts.append(_build_synthesis_prompt(state))
    parts.append(
        f"\nDraft to return with coverage/category fixes:\n--- DRAFT ---\n{draft}\n"
        "--- END DRAFT ---"
    )
    return "\n".join(parts)


def _build_citation_repair_prompt(state: GraphState, draft: str) -> str:
    registry = _build_citation_registry(state)
    parts = [f"User question:\n{state.user_input}\n", "Global Citation Registry:\n"]
    for entry in registry:
        source = entry.source
        if source.url:
            parts.append(f"[{entry.number}]({source.url}) — {source.name}\n")
        else:
            parts.append(f"[{entry.number}] — {source.name} (uploaded document)\n")

    parts.append("\nAgent evidence with its collected source citations:\n")
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
    """Render only sources cited in the body, in first-citation order."""
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

    return ""


def _build_citation_registry(state: GraphState) -> list[CitationEntry]:
    """Collect sources actually propagated through Agent reports."""
    entries: list[CitationEntry] = []
    seen: set[str] = set()
    sources = [
        source for report in state.reports.values() for source in report.sources if source.url
    ]
    for source in sources:
        if not source.url:
            continue
        key = canonical_source_url(source.url)
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
