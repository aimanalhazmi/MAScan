from pathlib import Path

import pytest

from mascan.eval.gold_standard import (
    PESTEL_HEADINGS,
    load_gold_standard,
    prompt_pack,
    validate_gold_standard_coverage,
)


def test_gold_standard_loads_all_cases():
    dataset = load_gold_standard()

    assert len(dataset.cases) == 25
    assert dataset.by_id("2007_1_SHELL").case_title
    assert dataset.by_id("2021_2_Unilever").case_subject.startswith("Celcom")


def test_gold_standard_cases_have_required_targets():
    dataset = load_gold_standard()

    for case in dataset.cases:
        assert Path(case.source_pdf).exists()
        assert case.prompt
        assert case.expected_output.strategic_implications
        assert len(case.gold_claims) >= 5
        assert len(case.category_targets) >= 6
        assert case.validation_notes.source_anchors
        assert case.validation_notes.reread_justification


def test_gold_standard_covers_pdf_inventory_exactly():
    dataset = load_gold_standard()

    report = validate_gold_standard_coverage(dataset)

    assert report.is_valid
    assert report.case_count == 25
    assert report.pdf_count == 25
    assert len(report.source_pdfs) == 25
    assert {Path(path).name for path in report.source_pdfs} == {
        Path(path).name for path in report.inventory_pdfs
    }


def test_gold_standard_coverage_report_flags_missing_paper_links():
    dataset = load_gold_standard().model_copy(deep=True)
    dataset.cases[0].source_pdf = "eval_papers/missing.pdf"

    report = validate_gold_standard_coverage(dataset)
    codes = {issue.code for issue in report.issues}

    assert not report.is_valid
    assert "source_pdf_missing_from_inventory" in codes
    assert "paper_missing_from_dataset" in codes


def test_gold_standard_coverage_report_flags_prompt_template_drift():
    dataset = load_gold_standard().model_copy(deep=True)
    dataset.cases[0].prompt = "Analyze the market."

    report = validate_gold_standard_coverage(dataset)

    assert not report.is_valid
    assert any(
        issue.code == "prompt_template_mismatch"
        and issue.case_id == dataset.cases[0].case_id
        for issue in report.issues
    )


def test_gold_standard_category_targets_cover_each_pestel_bucket():
    dataset = load_gold_standard()

    for case in dataset.cases:
        categories = {target.correct_category for target in case.category_targets}

        assert categories == set(PESTEL_HEADINGS), case.case_id


def test_prompt_pack_excludes_expected_answers():
    dataset = load_gold_standard()
    prompts = prompt_pack(dataset)

    assert len(prompts) == 25
    assert set(prompts[0]) == {"case_id", "case_title", "source_pdf", "prompt"}
    assert "expected_output" not in prompts[0]


def test_unknown_case_raises_key_error():
    dataset = load_gold_standard()

    with pytest.raises(KeyError):
        dataset.by_id("missing")
