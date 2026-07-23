from typing import Any

from mascan.contracts.planning import AgentAssignment
from mascan.contracts.reports import AgentReport, Source
from mascan.orchestrator.state import GraphState
from mascan.orchestrator.synthesizer import (
    SYNTHESIZER_SYSTEM_PROMPT,
    _build_citation_registry,
    _build_synthesis_prompt,
    _extract_cited_source_numbers,
    _extract_html_source_numbers,
    _needs_citation_repair,
    _normalize_citation_links,
    _render_sources_section,
    _renumber_all_citations,
    _renumber_citation_links,
    _strip_draft_wrappers,
    _synthesizer_node,
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


def test_synthesis_prompt_keeps_planner_coverage_checklist() -> None:
    state = make_state().model_copy(
        update={
            "plan": {
                "political": AgentAssignment(
                    agent_name="political",
                    objective_context="Assess policy exposure.",
                    tasks=["Assess industrial policy."],
                    salient_factors=["EU industrial subsidies"],
                )
            }
        }
    )

    prompt = _build_synthesis_prompt(state)

    assert "Coverage checklist" in prompt
    assert "EU industrial subsidies" in prompt


def test_full_pestel_draft_uses_only_initial_synthesis_call(mocker: Any) -> None:
    draft = """\
## Political
- Policy evidence [1](https://example.com/a).
## Economic
- Cost evidence.
## Social
- Labor evidence.
## Technological
- Technology evidence.
## Environmental
- Emissions evidence.
## Legal
- Regulation evidence.
## Strategic implications
- Prioritize the supported response.
"""
    model = mocker.Mock()
    model.invoke.return_value = mocker.Mock(content=draft)
    mocker.patch(
        "mascan.orchestrator.synthesizer.get_settings"
    ).return_value.openai_model_default = "test-model"
    get_model = mocker.patch(
        "mascan.orchestrator.synthesizer.get_chat_model",
        return_value=model,
    )

    result = _synthesizer_node(make_state())

    get_model.assert_called_once_with(
        model="test-model",
        temperature=0.3,
        max_tokens=4000,
    )
    model.invoke.assert_called_once()
    assert "--- DRAFT ---" not in result["final_summary"]
    assert "--- END DRAFT ---" not in result["final_summary"]
    assert "--- DRAFT ---" not in result["final_markdown"]
    assert "--- END DRAFT ---" not in result["final_markdown"]


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
        "Policy changed [1](https://example.com/b). Growth rose [2](https://example.com/a)."
    )
    assert _extract_cited_source_numbers(renumbered) == [1, 2]


def test_sources_follow_body_citation_order_and_remove_duplicates() -> None:
    summary = (
        "B changed [1](https://example.com/b). "
        "A changed [2](https://example.com/a). "
        "B again [1](https://example.com/b)."
    )

    assert _render_sources_section(make_state(), summary) == (
        "1. [Source B](https://example.com/b)\n2. [Source A](https://example.com/a)"
    )


def test_sources_do_not_fall_back_to_uncited_registry_entries() -> None:
    assert _render_sources_section(make_state(), "No numbered citations") == ""
    assert _render_sources_section(GraphState(user_input="Question"), "Summary") == ""


def test_uploaded_factsheet_reaches_registry_through_agent_report() -> None:
    upload = Source(
        name="EVONIK Analyst & Investor Factsheet Q1 2026.pdf",
        url=("/rag/files/EVONIK%20Analyst%20%26%20Investor%20Factsheet%20Q1%202026.pdf"),
    )
    state = make_state()
    economics = state.reports["economics"].model_copy(
        update={
            "findings": f"Adjusted EBITDA increased [Factsheet]({upload.url}).",
            "sources": [*state.reports["economics"].sources, upload],
        }
    )
    state = state.model_copy(update={"reports": {**state.reports, "economics": economics}})
    registry = _build_citation_registry(state)
    prompt = _build_synthesis_prompt(state)

    assert len(registry) == 3
    assert all(entry.source.url for entry in registry)
    assert [entry.source.url for entry in registry] == [
        "https://example.com/a",
        upload.url,
        "https://example.com/b",
    ]
    assert "Adjusted EBITDA" in prompt
    assert _needs_citation_repair(state, "Web fact [1](https://example.com/a).") is False


def test_raw_rag_evidence_is_not_injected_directly_into_synthesizer() -> None:
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
    assert "First-page evidence" not in prompt
    assert "Third-page evidence" not in prompt


def test_web_and_upload_sources_are_renumbered_by_first_body_appearance() -> None:
    state = make_state()
    upload = Source(name="factsheet.pdf", url="/rag/files/factsheet.pdf")
    economics = state.reports["economics"].model_copy(
        update={"sources": [*state.reports["economics"].sources, upload]}
    )
    state = state.model_copy(update={"reports": {**state.reports, "economics": economics}})
    draft = (
        "Company fact [3](/rag/files/factsheet.pdf). "
        "Market fact [1](https://example.com/a). "
        "Company fact again [3](/rag/files/factsheet.pdf)."
    )

    summary, sources = _renumber_all_citations(state, draft)

    assert summary == (
        "Company fact [1](/rag/files/factsheet.pdf). "
        "Market fact [2](https://example.com/a). "
        "Company fact again [1](/rag/files/factsheet.pdf)."
    )
    assert [source.url for source in sources] == [
        "/rag/files/factsheet.pdf",
        "https://example.com/a",
    ]
    assert _render_sources_section(state, summary, sources) == (
        "1. [factsheet.pdf](/rag/files/factsheet.pdf)\n2. [Source A](https://example.com/a)"
    )


def test_synthesizer_preserves_agent_citations_without_attachment_special_case() -> None:
    assert "Preserve relevant inline citations" in SYNTHESIZER_SYSTEM_PROMPT
    assert "including `/rag/files/...` links" in SYNTHESIZER_SYSTEM_PROMPT
    assert "explicitly asks to use an attached" not in SYNTHESIZER_SYSTEM_PROMPT


def test_citation_repair_draft_wrappers_are_removed() -> None:
    repaired = "--- DRAFT ---\nReport body [1](https://example.com).\n--- END DRAFT ---"

    assert _strip_draft_wrappers(repaired) == "Report body [1](https://example.com)."
