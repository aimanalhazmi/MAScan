from mascan.contracts.reports import AgentReport, Source
from mascan.orchestrator.state import GraphState
from mascan.orchestrator.synthesizer import (
    SYNTHESIZER_SYSTEM_PROMPT,
    _build_citation_registry,
    _build_synthesis_prompt,
    _extract_cited_source_numbers,
    _extract_html_source_numbers,
    _normalize_citation_links,
    _render_sources_section,
    _renumber_citation_links,
)


def make_report(name: str, sources: list[Source]) -> AgentReport:
    return AgentReport(
        agent_name=name,
        findings=f"{name} findings",
        sources=sources,
        confidence=0.8,
        rendered_markdown=f"## {name}",
    )


def make_state() -> GraphState:
    return GraphState(
        user_input="Assess the market",
        reports={
            "economics": make_report(
                "economics",
                [
                    Source(name="Source A", url="https://example.com/a"),
                    Source(name="Offline source"),
                ],
            ),
            "political": make_report(
                "political",
                [
                    Source(name="Source A duplicate", url="https://example.com/a"),
                    Source(name="Source B", url="https://example.com/b"),
                ],
            ),
        },
    )


def test_registry_deduplicates_urls_and_keeps_stable_numbers() -> None:
    registry = _build_citation_registry(make_state())

    assert [(entry.number, entry.source.name, entry.source.url) for entry in registry] == [
        (1, "Source A", "https://example.com/a"),
        (2, "Source B", "https://example.com/b"),
    ]


def test_synthesis_prompt_contains_registry_and_no_url_guidance() -> None:
    prompt = _build_synthesis_prompt(make_state())

    assert "[1] Source A - https://example.com/a" in prompt
    assert "[2] Source B - https://example.com/b" in prompt
    assert "Offline source (no URL)" in prompt

    empty_prompt = _build_synthesis_prompt(GraphState(user_input="Question"))
    assert "No URL-backed sources are available" in empty_prompt


def test_legacy_html_citations_are_normalized_and_renumbered() -> None:
    state = make_state()
    legacy = (
        'Policy changed <sup><a href="#source-2">[2]</a></sup>. '
        'Growth rose <sup><a href="#source-1">[1]</a></sup>.'
    )

    normalized = _normalize_citation_links(state, legacy)
    assert "[2](https://example.com/b)" in normalized
    assert "[1](https://example.com/a)" in normalized
    assert _extract_html_source_numbers(legacy) == [2, 1]

    renumbered = _renumber_citation_links(normalized)
    assert renumbered == (
        "Policy changed [1](https://example.com/b). "
        "Growth rose [2](https://example.com/a)."
    )
    assert _extract_cited_source_numbers(renumbered) == [1, 2]


def test_sources_follow_body_citation_order_and_remove_duplicates() -> None:
    summary = (
        "B changed [1](https://example.com/b). "
        "A changed [2](https://example.com/a). "
        "B again [1](https://example.com/b)."
    )

    assert _render_sources_section(make_state(), summary) == (
        "1. [Source B](https://example.com/b)\n"
        "2. [Source A](https://example.com/a)"
    )


def test_sources_fall_back_to_registry_when_body_has_no_citations() -> None:
    assert _render_sources_section(make_state(), "No numbered citations") == (
        "1. [Source A](https://example.com/a)\n"
        "2. [Source B](https://example.com/b)"
    )
    assert _render_sources_section(GraphState(user_input="Question"), "Summary") == ""


def test_uploaded_factsheet_is_background_but_not_a_final_source() -> None:
    state = make_state().model_copy(
        update={
            "rag_evidence": [
                {
                    "content": "Adjusted EBITDA increased in Q1 2026.",
                    "citation": {
                        "document": "EVONIK Analyst & Investor Factsheet Q1 2026.pdf",
                        "page": 4,
                    },
                    "score": 0.91,
                }
            ]
        }
    )
    registry = _build_citation_registry(state)
    prompt = _build_synthesis_prompt(state)

    assert len(registry) == 2
    assert all(entry.source.url for entry in registry)
    assert "Adjusted EBITDA" in prompt
    assert "background context only; do not cite it" in prompt
    assert "Factsheet" not in _render_sources_section(state, "No citations")


def test_uploaded_factsheet_pages_share_one_background_evidence_block() -> None:
    state = GraphState(
        user_input="Assess EVONIK",
        rag_evidence=[
            {
                "content": "First-page evidence.",
                "citation": {"document": "factsheet.pdf", "page": 1},
            },
            {
                "content": "Third-page evidence.",
                "citation": {"document": "factsheet.pdf", "page": 3},
            },
        ],
    )

    prompt = _build_synthesis_prompt(state)

    assert _build_citation_registry(state) == []
    assert prompt.count("factsheet.pdf") == 1
    assert "[Page 1]" in prompt
    assert "[Page 3]" in prompt


def test_synthesizer_requires_web_sources_for_external_pestel_claims() -> None:
    assert "company-specific" in SYNTHESIZER_SYSTEM_PROMPT
    assert "external PESTEL" in SYNTHESIZER_SYSTEM_PROMPT
    assert "must include the relevant URL-backed citations" in SYNTHESIZER_SYSTEM_PROMPT
