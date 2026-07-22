from contextvars import ContextVar
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mascan.contracts.tools import ToolResult
from mascan.contracts.validation import (
    CitationCheck,
    ValidationCitation,
    ValidationIssue,
    ValidationReport,
    ValidationSummary,
)
from mascan.orchestrator.attribution import parse_attribution_document
from mascan.orchestrator.graph import state_to_report
from mascan.orchestrator.state import GraphState
from mascan.orchestrator.validator import (
    FACT_CHECK_SYSTEM_PROMPT,
    RELEVANCE_SYSTEM_PROMPT,
    FactCheckJudgeResult,
    RelevanceJudgeResult,
    SourceCheck,
    evaluate_citation_pairs,
    evaluate_fetched_pair,
    fetch_cited_sources,
    issues_from_checks,
    render_validation_markdown,
    run_validation,
    summarize_checks,
    validation_status,
    validator_node,
)


class FakeModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    def invoke(self, messages: object) -> object:
        self.calls += 1
        return self.result


def make_state(
    markdown: str = "## Summary\n\nGDP grew [1](https://example.com/gdp).",
) -> GraphState:
    return GraphState(
        user_input="Assess growth",
        final_summary="GDP grew.",
        final_markdown=markdown,
    )


def make_attribution():
    document = parse_attribution_document("## Summary\n\nGDP grew [1](https://example.com/gdp).")
    return document.attributions[0], document.citations[0]


def test_prompts_define_separate_staged_judges() -> None:
    assert "partially_relevant" in RELEVANCE_SYSTEM_PROMPT
    assert "Fact Check" not in RELEVANCE_SYSTEM_PROMPT
    assert "not whether the source proves the claim" in RELEVANCE_SYSTEM_PROMPT
    assert "partially_supported" in FACT_CHECK_SYSTEM_PROMPT
    assert "Missing support is not contradiction" in FACT_CHECK_SYSTEM_PROMPT
    assert "every distinct material factual premise" in FACT_CHECK_SYSTEM_PROMPT


@pytest.mark.parametrize("subtype", ["partially_relevant", "unrelated", "uncertain"])
def test_relevance_issue_stops_before_fact_check(subtype: str) -> None:
    attribution, citation = make_attribution()
    relevance = FakeModel(
        RelevanceJudgeResult(relevant_content=subtype, explanation="Scope does not match.")
    )
    fact_check = FakeModel(
        FactCheckJudgeResult(fact_check="supported", explanation="Should not run.")
    )

    check = evaluate_fetched_pair(
        relevance,
        fact_check,
        attribution,
        citation,
        {"markdown": "Source evidence"},
    )
    issue = issues_from_checks([check])[0]

    assert check.status == "issue"
    assert check.stopped_after == "relevant_content"
    assert check.fact_check is None
    assert fact_check.calls == 0
    assert issue.category == "relevant_content"
    assert issue.subtype == subtype


@pytest.mark.parametrize(
    "subtype",
    ["partially_supported", "unsupported", "contradicted", "uncertain"],
)
def test_fact_check_runs_only_after_relevance_passes(subtype: str) -> None:
    attribution, citation = make_attribution()
    relevance = FakeModel(
        RelevanceJudgeResult(relevant_content="relevant", explanation="Same topic.")
    )
    fact_check = FakeModel(
        FactCheckJudgeResult(fact_check=subtype, explanation="Fact check failed.")
    )

    check = evaluate_fetched_pair(
        relevance,
        fact_check,
        attribution,
        citation,
        {"markdown": "Source evidence"},
    )
    issue = issues_from_checks([check])[0]

    assert check.status == "issue"
    assert check.relevant_content == "relevant"
    assert check.fact_check == subtype
    assert check.stopped_after == "fact_check"
    assert fact_check.calls == 1
    assert issue.category == "fact_check"
    assert issue.subtype == subtype


def test_supported_pair_passes_without_an_issue() -> None:
    attribution, citation = make_attribution()
    check = evaluate_fetched_pair(
        FakeModel(RelevanceJudgeResult(relevant_content="relevant", explanation="Same topic.")),
        FakeModel(FactCheckJudgeResult(fact_check="supported", explanation="Claim matches.")),
        attribution,
        citation,
        {"markdown": "GDP grew."},
    )

    assert check.status == "passed"
    assert issues_from_checks([check]) == []


def test_inaccessible_source_stops_without_creating_models() -> None:
    document = parse_attribution_document("## Summary\n\nGDP grew [1](https://example.com/gdp).")
    checks = evaluate_citation_pairs(
        document,
        [
            SourceCheck(
                citation_numbers=[1],
                requested_url="https://example.com/gdp",
                status="inaccessible",
                checked_at="2026-07-18T00:00:00+00:00",
                error="HTTP 404",
            )
        ],
        {},
    )

    assert checks[0].stopped_after == "link_works"
    assert checks[0].relevant_content is None
    assert checks[0].fact_check is None
    issue = issues_from_checks(checks)[0]
    assert issue.category == "inaccessible_source"
    assert issue.subtype is None


def test_operational_source_failure_is_not_an_inaccessible_issue() -> None:
    document = parse_attribution_document("## Summary\n\nGDP grew [1](https://example.com/gdp).")
    checks = evaluate_citation_pairs(
        document,
        [
            SourceCheck(
                citation_numbers=[1],
                requested_url="https://example.com/gdp",
                status="failed",
                checked_at="2026-07-18T00:00:00+00:00",
                error="document_antibot",
            )
        ],
        {},
    )

    assert checks[0].status == "failed"
    assert checks[0].stopped_after == "link_works"
    assert issues_from_checks(checks) == []


def test_repeated_source_is_validated_once_per_claim_citation_pair() -> None:
    document = parse_attribution_document(
        "## Summary\n\n"
        "First claim [1](https://example.com/source).\n\n"
        "Second claim [1](https://example.com/source)."
    )
    canonical_url = document.citations[0].canonical_url
    relevance = FakeModel(
        RelevanceJudgeResult(relevant_content="unrelated", explanation="Wrong topic.")
    )
    fact_check = FakeModel(
        FactCheckJudgeResult(fact_check="supported", explanation="Should not run.")
    )

    checks = evaluate_citation_pairs(
        document,
        [
            SourceCheck(
                citation_numbers=[1],
                requested_url="https://example.com/source",
                status="fetched",
                checked_at="2026-07-20T00:00:00+00:00",
            )
        ],
        {canonical_url: {"markdown": "Source evidence"}},
        relevance_model=relevance,
        fact_check_model=fact_check,
    )

    assert [check.claim for check in checks] == ["First claim.", "Second claim."]
    assert all(check.stopped_after == "relevant_content" for check in checks)
    assert all(check.fact_check is None for check in checks)
    assert relevance.calls == 2
    assert fact_check.calls == 0
    assert len(issues_from_checks(checks)) == 2


def test_issue_schema_rejects_category_subtype_mismatch() -> None:
    with pytest.raises(ValidationError):
        ValidationIssue(
            category="relevant_content",
            subtype="unsupported",
            claim="Claim",
            passage="Claim [1](https://example.com).",
            citation=ValidationCitation(number=1, url="https://example.com"),
            explanation="Wrong subtype family.",
        )


def test_source_fetch_deduplicates_urls_without_a_cap() -> None:
    paragraphs = [
        f"Claim {number} [{number}](https://example.com/{number})." for number in range(1, 13)
    ]
    paragraphs.append("Repeated source [13](https://example.com/1).")
    document = parse_attribution_document("## Summary\n\n" + "\n\n".join(paragraphs))

    class FakeSourceFetch:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def run(self, url: str) -> ToolResult[dict[str, object]]:
            self.calls.append(url)
            return ToolResult(
                success=True,
                source="source_fetch:test",
                data={
                    "final_url": url,
                    "title": url,
                    "markdown": "Evidence",
                    "checked_at": "2026-07-18T00:00:00+00:00",
                },
            )

    tool = FakeSourceFetch()
    checks, fetched = fetch_cited_sources(document, tool=tool)

    assert len(tool.calls) == 12
    assert len(set(tool.calls)) == 12
    assert len(checks) == 12
    assert len(fetched) == 12


def test_uploaded_file_uses_rag_evidence_without_source_fetch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_name = "EVONIK Q1 2026.pdf"
    (tmp_path / document_name).write_bytes(b"%PDF-1.4 retained original")
    monkeypatch.setattr(
        "mascan.orchestrator.validator.get_settings",
        lambda: SimpleNamespace(rag_upload_dir=str(tmp_path)),
    )
    state = GraphState(
        user_input="Use the attached EVONIK factsheet",
        final_markdown=(
            "## Summary\n\nAdjusted EBITDA increased [1](/rag/files/EVONIK%20Q1%202026.pdf)."
        ),
        rag_evidence=[
            {
                "content": "Adjusted EBITDA increased in Q1 2026.",
                "citation": {"document": document_name, "page": 4},
            }
        ],
    )
    parsed = parse_attribution_document(state.final_markdown)

    checks, fetched = fetch_cited_sources(parsed, state=state)
    evaluated = evaluate_citation_pairs(
        parsed,
        checks,
        fetched,
        relevance_model=FakeModel(
            RelevanceJudgeResult(relevant_content="relevant", explanation="Same topic.")
        ),
        fact_check_model=FakeModel(
            FactCheckJudgeResult(fact_check="supported", explanation="Claim matches.")
        ),
    )

    assert checks[0].status == "fetched"
    assert "[Page 4]" in fetched[parsed.citations[0].canonical_url]["markdown"]
    assert evaluated[0].status == "passed"


def test_uploaded_file_is_inaccessible_only_when_original_is_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_name = "factsheet.pdf"
    monkeypatch.setattr(
        "mascan.orchestrator.validator.get_settings",
        lambda: SimpleNamespace(rag_upload_dir=str(tmp_path)),
    )
    state = GraphState(
        user_input="Use the attachment",
        final_markdown=("## Summary\n\nCompany fact [1](/rag/files/factsheet.pdf)."),
    )
    parsed = parse_attribution_document(state.final_markdown)

    missing_checks, missing_fetched = fetch_cited_sources(parsed, state=state)
    missing_evaluated = evaluate_citation_pairs(
        parsed,
        missing_checks,
        missing_fetched,
    )

    assert missing_checks[0].status == "inaccessible"
    assert issues_from_checks(missing_evaluated)[0].category == "inaccessible_source"

    (tmp_path / document_name).write_bytes(b"%PDF-1.4 retained original")
    no_evidence_checks, no_evidence_fetched = fetch_cited_sources(parsed, state=state)
    no_evidence_evaluated = evaluate_citation_pairs(
        parsed,
        no_evidence_checks,
        no_evidence_fetched,
    )

    assert no_evidence_checks[0].status == "failed"
    assert issues_from_checks(no_evidence_evaluated) == []


def test_summary_and_status_preserve_partial_failures() -> None:
    citation = ValidationCitation(number=1, url="https://example.com")
    checks = [
        CitationCheck(
            status="passed",
            claim="Claim",
            passage="Passage",
            citation=citation,
            link_works=True,
            relevant_content="relevant",
            fact_check="supported",
            stopped_after="fact_check",
            explanation="Supported.",
        ),
        CitationCheck(
            status="failed",
            claim="Other claim",
            passage="Other passage",
            citation=citation,
            link_works=True,
            stopped_after="relevant_content",
            explanation="Judge failed.",
            error="RuntimeError: unavailable",
        ),
    ]

    summary = summarize_checks(checks)

    assert summary.model_dump() == {"total": 2, "passed": 1, "issues": 0, "failed": 1}
    assert validation_status(summary) == "warnings"
    assert validation_status(ValidationSummary(total=1, passed=0, issues=0, failed=1)) == (
        "failed_to_validate"
    )


def test_no_citations_returns_an_explicit_empty_result() -> None:
    report = run_validation(make_state("## Summary\n\nNo citation."))

    assert report.status == "passed"
    assert report.summary.total == 0
    assert "No citation pairs" in report.markdown


def test_validation_markdown_renders_new_category_and_subtype() -> None:
    issue = ValidationIssue(
        category="fact_check",
        subtype="contradicted",
        claim="GDP fell.",
        passage="GDP fell [1](https://example.com/gdp).",
        citation=ValidationCitation(number=1, url="https://example.com/gdp"),
        explanation="The source reports growth.",
    )
    report = ValidationReport(
        status="warnings",
        summary=ValidationSummary(total=1, passed=0, issues=1, failed=0),
        issues=[issue],
    )

    rendered = render_validation_markdown(report)

    assert "## Citation Validation" in rendered
    assert "fact_check / contradicted" in rendered
    assert "[1](https://example.com/gdp)" in rendered


def test_validator_node_preserves_report_and_emits_one_validation_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = ValidationReport(
        status="passed",
        summary=ValidationSummary(total=0, passed=0, issues=0, failed=0),
        markdown="## Citation Validation",
    )
    monkeypatch.setattr("mascan.orchestrator.validator.run_validation", lambda state: report)

    state = make_state()
    update = validator_node(state)

    assert update["final_markdown"] == state.final_markdown
    assert update["validation"]["status"] == "passed"
    assert "validation_status" not in update
    assert "Citation Validation" not in update["final_markdown"]


def test_validator_failure_does_not_discard_final_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(state: GraphState) -> ValidationReport:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("mascan.orchestrator.validator.run_validation", fail)
    state = make_state()

    update = validator_node(state)

    assert update["final_markdown"] == state.final_markdown
    assert update["validation"]["status"] == "failed_to_validate"
    assert update["validation"]["error"] == "RuntimeError: model unavailable"


def test_citation_workers_inherit_the_validator_measurement_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = ContextVar("metrics_owner", default="missing")
    owner.set("validator")
    seen_owners: list[str] = []

    def judge(relevance_model, fact_check_model, attribution, citation, source):
        seen_owners.append(owner.get())
        return CitationCheck(
            status="passed",
            claim=attribution.claim,
            passage=attribution.passage,
            citation=ValidationCitation(number=citation.number, url=citation.url),
            link_works=True,
            relevant_content="relevant",
            fact_check="supported",
            stopped_after="fact_check",
            explanation="Supported by the excerpt.",
        )

    monkeypatch.setattr("mascan.orchestrator.validator.evaluate_fetched_pair", judge)
    document = parse_attribution_document("## Summary\n\nGDP grew [1](https://example.com/gdp).")

    checks = evaluate_citation_pairs(
        document,
        [
            SourceCheck(
                citation_numbers=[1],
                requested_url="https://example.com/gdp",
                final_url="https://example.com/gdp",
                title="GDP release",
                status="fetched",
                checked_at="2026-07-18T00:00:00+00:00",
            )
        ],
        {
            "https://example.com/gdp": {
                "title": "GDP release",
                "final_url": "https://example.com/gdp",
                "markdown": "GDP grew.",
            }
        },
        relevance_model=object(),
        fact_check_model=object(),
    )

    assert [check.status for check in checks] == ["passed"]
    assert seen_owners == ["validator"]


def test_final_report_keeps_validation_inside_metadata() -> None:
    validation = ValidationReport(
        status="passed",
        summary=ValidationSummary(total=0, passed=0, issues=0, failed=0),
    )
    report = state_to_report(
        {
            "user_input": "Question",
            "final_summary": "Summary",
            "final_markdown": "# Final Report",
            "validation": validation,
        }
    )

    assert report.metadata["validation"]["status"] == "passed"
    assert not hasattr(report, "validation")
