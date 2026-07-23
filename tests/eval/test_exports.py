from mascan.eval.exports import (
    GOLD_STANDARD_MANIFEST_CSV_FIELDS,
    PROMPT_PACK_CSV_FIELDS,
    gold_standard_manifest_csv_rows,
    gold_standard_manifest_payload,
    prompt_pack_csv_rows,
    render_gold_judge_rubric_markdown,
    render_gold_standard_manifest_markdown,
    render_gold_standard_validation_report,
    render_prompt_pack_markdown,
)
from mascan.eval.gold_standard import load_gold_standard


def test_prompt_pack_exports_markdown_and_csv_rows():
    dataset = load_gold_standard()

    markdown = render_prompt_pack_markdown(dataset)
    rows = prompt_pack_csv_rows(dataset)

    assert "# Gold-Standard PESTEL Prompt Pack" in markdown
    assert "2007_1_SHELL" in markdown
    assert len(rows) == 25
    assert set(rows[0]) == set(PROMPT_PACK_CSV_FIELDS)


def test_gold_standard_validation_report_includes_reread_evidence():
    dataset = load_gold_standard()

    markdown = render_gold_standard_validation_report(dataset)

    assert "# Gold-Standard Dataset Validation Report" in markdown
    assert "- Cases: 25" in markdown
    assert "- PDF papers in inventory: 25" in markdown
    assert "- Dataset/PDF coverage exact: yes" in markdown
    assert "- Expected output sections complete: yes" in markdown
    assert "- Categorization targets cover all PESTEL buckets per case: yes" in markdown
    assert "## 2007_1_SHELL" in markdown
    assert "### Reread Justification" in markdown


def test_gold_standard_manifest_exports_case_hashes_and_counts():
    dataset = load_gold_standard()

    payload = gold_standard_manifest_payload(dataset)
    rows = gold_standard_manifest_csv_rows(dataset)
    markdown = render_gold_standard_manifest_markdown(dataset)

    assert payload["case_count"] == 25
    assert payload["pdf_count"] == 25
    assert payload["coverage_valid"] is True
    assert len(str(payload["dataset_sha256"])) == 64
    assert len(payload["cases"]) == 25
    first = payload["cases"][0]
    assert first["case_id"] == "2007_1_SHELL"
    assert len(first["prompt_sha256"]) == 64
    assert len(first["expected_output_sha256"]) == 64
    assert first["gold_claim_count"] >= 5
    assert first["category_target_count"] == 6
    assert first["expected_sections_complete"] is True
    assert first["category_targets_cover_all_buckets"] is True
    assert len(rows) == 25
    assert set(rows[0]) == set(GOLD_STANDARD_MANIFEST_CSV_FIELDS)
    assert "# Gold-Standard Dataset Freeze Manifest" in markdown
    assert "Dataset SHA-256" in markdown
    assert "2007_1_SHELL" in markdown


def test_gold_judge_rubric_export_includes_prompt_schema_and_sample_case():
    case = load_gold_standard().by_id("2007_1_SHELL")

    markdown = render_gold_judge_rubric_markdown(
        sample_case=case,
        sample_response_text="Political: EU policy affects Shell.",
    )

    assert "# Gold-Standard PESTEL Judge Rubric" in markdown
    assert "judge_prompt_sha256" in markdown
    assert "judge_schema_sha256" in markdown
    assert "Enumerate every distinct causal relationship" in markdown
    assert "`response_claim_scores`" in markdown
    assert '"category_judgments"' in markdown
    assert "## Sample Case-Specific User Prompt" in markdown
    assert "2007_1_SHELL" in markdown
    assert "Political: EU policy affects Shell." in markdown
