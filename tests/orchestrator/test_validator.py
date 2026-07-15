from contextvars import ContextVar

import pytest

from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.orchestrator.attribution import parse_attribution_document
from mascan.orchestrator.graph import state_to_report
from mascan.orchestrator.state import GraphState
from mascan.orchestrator.validator import (
    CitationEvaluation,
    SourceCheck,
    ValidationIssue,
    ValidationResult,
    build_validation_prompt,
    evaluate_citation_pairs,
    fetch_cited_sources,
    issues_from_citation_evaluations,
    render_issue_citations,
    render_validation_markdown,
    sanitize_report_issues,
    validator_node,
)


def make_state() -> GraphState:
    report = AgentReport(
        agent_name="economics",
        findings="GDP grew according to the supplied release.",
        sources=[Source(name="GDP release", url="https://example.com/gdp")],
        confidence=0.9,
        rendered_markdown="GDP grew.",
    )
    return GraphState(
        user_input="Assess growth",
        reports={"economics": report},
        failures={"social": "source unavailable"},
        final_summary="GDP grew [1](https://example.com/gdp).",
        final_markdown="# Final Report\n\nGDP grew [1](https://example.com/gdp).",
    )


def test_validation_markdown_passes_without_issues() -> None:
    rendered = render_validation_markdown(
        ValidationResult(issues=[], overall_note="Claims match the supplied evidence.")
    )

    assert "**Status:** passed" in rendered
    assert "No obvious factual issues" in rendered


def test_validation_markdown_renders_issue_citations() -> None:
    issue = ValidationIssue(
        category="source_mismatch",
        severity="high",
        claim="GDP fell [1](https://example.com/gdp).",
        explanation="The agent reported growth.",
        relevant_agents=["economics"],
    )
    rendered = render_validation_markdown(
        ValidationResult(issues=[issue], overall_note="One contradiction was found.")
    )

    assert "**Status:** warnings" in rendered
    assert "[1](https://example.com/gdp)" in rendered
    assert "Relevant evidence: economics" in rendered


def test_citation_gap_is_explicitly_marked_missing() -> None:
    issue = ValidationIssue(
        category="citation_gap",
        severity="medium",
        claim="GDP will double next year.",
        explanation="No source number is attached.",
    )

    assert render_issue_citations(issue) == "missing"


def test_validation_prompt_contains_report_sources_and_failures() -> None:
    prompt = build_validation_prompt(make_state())

    assert "Final report to validate" in prompt
    assert "GDP grew [1](https://example.com/gdp)" in prompt
    assert "GDP release: https://example.com/gdp" in prompt
    assert "social: source unavailable" in prompt


def test_validator_keeps_fact_check_separate_and_exposes_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ValidationResult(issues=[], overall_note="Evidence is consistent.")
    monkeypatch.setattr("mascan.orchestrator.validator.run_validation", lambda state: result)

    update = validator_node(make_state())

    assert update["validation_status"] == "passed"
    assert update["validation_payload"]["overall_note"] == "Evidence is consistent."
    assert update["final_markdown"] == make_state().final_markdown
    assert "## Fact Check" not in update["final_markdown"]
    assert "## Fact Check" in update["validation_markdown"]


def test_validator_failure_does_not_discard_final_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(state: GraphState) -> ValidationResult:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("mascan.orchestrator.validator.run_validation", fail)

    update = validator_node(make_state())

    assert update["final_markdown"] == make_state().final_markdown
    assert "**Status:** failed to validate" not in update["final_markdown"]
    assert "**Status:** failed to validate" in update["validation_markdown"]
    assert update["validation_payload"]["error"] == "RuntimeError: model unavailable"


def test_citation_workers_inherit_the_validator_measurement_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = ContextVar("metrics_owner", default="missing")
    owner.set("validator")
    seen_owners: list[str] = []

    def judge(model, attribution, citation, source, excerpt):
        seen_owners.append(owner.get())
        return CitationEvaluation(
            claim=attribution.claim,
            passage=attribution.passage,
            citation_number=citation.number,
            url=citation.url,
            link_works=True,
            relevant_content="relevant",
            fact_check="supported",
            explanation="Supported by the excerpt.",
            status="completed",
        )

    monkeypatch.setattr(
        "mascan.orchestrator.validator.get_citation_validation_model",
        lambda: object(),
    )
    monkeypatch.setattr("mascan.orchestrator.validator._judge_pair_with_retries", judge)
    document = parse_attribution_document(
        "GDP grew [1](https://example.com/gdp)."
    )

    evaluations = evaluate_citation_pairs(
        document,
        [
            SourceCheck(
                citation_numbers=[1],
                requested_url="https://example.com/gdp",
                final_url="https://example.com/gdp",
                title="GDP release",
                status="fetched",
                checked_at="2026-07-15T00:00:00+00:00",
            )
        ],
        {
            "https://example.com/gdp": {
                "title": "GDP release",
                "final_url": "https://example.com/gdp",
                "markdown": "GDP grew.",
            }
        },
    )

    assert [evaluation.status for evaluation in evaluations] == ["completed"]
    assert seen_owners == ["validator"]


def test_final_report_metadata_contains_validation_payload() -> None:
    report = state_to_report(
        {
            "user_input": "Question",
            "final_summary": "Summary",
            "final_markdown": "# Final Report",
            "validation_payload": {"status": "passed", "issues": []},
        }
    )

    assert report.summary == "Summary"
    assert report.metadata == {"validation": {"status": "passed", "issues": []}}


def test_cited_passage_cannot_be_reported_as_citation_gap() -> None:
    document = parse_attribution_document(
        "## Summary\n\nThe EU targets ten million tonnes by 2030 [1](https://example.com/eu)."
    )
    issue = ValidationIssue(
        category="citation_gap",
        severity="high",
        claim="The EU targets ten million tonnes by 2030.",
        explanation="No citation was copied into the claim field.",
    )

    assert sanitize_report_issues([issue], document) == []


def test_pairwise_source_mismatch_becomes_structured_issue() -> None:
    evaluation = CitationEvaluation(
        claim="The EU target is ten million tonnes.",
        passage="The EU target is ten million tonnes [1](https://example.com/eu).",
        citation_number=1,
        url="https://example.com/eu",
        link_works=True,
        relevant_content="relevant",
        fact_check="unsupported",
        explanation="The page discusses hydrogen but does not state this target.",
        status="completed",
    )

    issues = issues_from_citation_evaluations([evaluation])

    assert len(issues) == 1
    assert issues[0].category == "source_mismatch"
    assert issues[0].severity == "medium"
    assert render_issue_citations(issues[0]) == "[1](https://example.com/eu)"


def test_pairwise_partial_support_is_a_low_severity_warning() -> None:
    evaluation = CitationEvaluation(
        claim="Energy costs rose and doubled.",
        passage="Energy costs rose and doubled [1](https://example.com/energy).",
        citation_number=1,
        url="https://example.com/energy",
        link_works=True,
        relevant_content="relevant",
        fact_check="partially_supported",
        explanation="The source reports an increase but does not support the doubling claim.",
        status="completed",
    )

    issues = issues_from_citation_evaluations([evaluation])

    assert len(issues) == 1
    assert issues[0].category == "source_mismatch"
    assert issues[0].severity == "low"


def test_citation_judge_prompt_separates_facts_from_recommendations() -> None:
    from mascan.orchestrator.validator import CITATION_JUDGE_SYSTEM_PROMPT

    assert "externally verifiable factual premises" in CITATION_JUDGE_SYSTEM_PROMPT
    assert "does not need to" in CITATION_JUDGE_SYSTEM_PROMPT
    assert "partially_supported" in CITATION_JUDGE_SYSTEM_PROMPT
    assert "Never use contradicted merely because support is missing" in CITATION_JUDGE_SYSTEM_PROMPT


def test_source_fetch_queue_has_no_ten_url_cap_and_deduplicates() -> None:
    paragraphs = [
        f"Claim {number} [{number}](https://example.com/{number})."
        for number in range(1, 13)
    ]
    paragraphs.append("Repeated source [13](https://example.com/1).")
    document = parse_attribution_document(
        "## Summary\n\n" + "\n\n".join(paragraphs)
    )

    class FakeSourceFetch:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(self, url: str) -> ToolResult[dict[str, object]]:
            self.calls.append(url)
            return ToolResult(
                success=True,
                source="source_fetch:test",
                data={
                    "requested_url": url,
                    "final_url": url,
                    "title": url,
                    "markdown": "Evidence",
                    "checked_at": "2026-07-14T00:00:00+00:00",
                },
            )

    tool = FakeSourceFetch()

    checks, fetched = fetch_cited_sources(document, tool=tool)

    assert len(tool.calls) == 12
    assert len(set(tool.calls)) == 12
    assert len(checks) == 12
    assert len(fetched) == 12


def test_fact_check_multidigit_items_use_commonmark_indentation() -> None:
    issues = [
        ValidationIssue(
            category="citation_gap",
            severity="medium",
            claim=f"Uncited claim {number}.",
            explanation="An important external fact needs evidence.",
        )
        for number in range(1, 11)
    ]

    rendered = render_validation_markdown(
        ValidationResult(issues=issues, overall_note="Review required.")
    )

    assert "9. **medium / citation_gap**\n   - Claim:" in rendered
    assert "10. **medium / citation_gap**\n    - Claim:" in rendered


def test_validation_result_accepts_any_number_of_issues() -> None:
    issues = [
        ValidationIssue(
            category="citation_gap",
            severity="low" if number < 5 else "high",
            claim=f"Claim {number}",
            explanation="Review required.",
        )
        for number in range(11)
    ]

    parsed = ValidationResult(issues=issues, overall_note="Review completed.")
    assert len(parsed.issues) == 11


def test_report_validator_prompt_does_not_require_citations_for_recommendations() -> None:
    from mascan.orchestrator.validator import VALIDATOR_SYSTEM_PROMPT

    assert "Strategic recommendations" in VALIDATOR_SYSTEM_PROMPT
    assert "Rough cost allocations" in VALIDATOR_SYSTEM_PROMPT
    assert "do not require their own citation" in VALIDATOR_SYSTEM_PROMPT


def test_uploaded_document_citation_is_not_sent_to_source_fetch() -> None:
    document = parse_attribution_document(
        "## Summary\n\nCompany revenue rose [1]. External fact [2](https://example.com/fact)."
    )

    class FakeSourceFetch:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(self, url: str) -> ToolResult[dict[str, object]]:
            self.calls.append(url)
            return ToolResult(
                success=True,
                source="source_fetch:test",
                data={
                    "requested_url": url,
                    "final_url": url,
                    "title": "Fact",
                    "markdown": "External evidence",
                    "checked_at": "2026-07-14T00:00:00+00:00",
                },
            )

    tool = FakeSourceFetch()
    checks, _ = fetch_cited_sources(document, tool=tool)

    assert tool.calls == ["https://example.com/fact"]
    assert len(checks) == 1
