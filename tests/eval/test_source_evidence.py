from mascan.eval.gold_standard import GoldStandardDataset
from mascan.eval.source_evidence import (
    SOURCE_EVIDENCE_CSV_FIELDS,
    render_source_evidence_markdown,
    source_evidence_csv_rows,
    validate_source_anchor_evidence,
)


def _dataset() -> GoldStandardDataset:
    return GoldStandardDataset.model_validate(
        {
            "schema_version": "1.0",
            "created_at": "2026-07-12",
            "purpose": "test",
            "generation_instruction_template": "prompt",
            "rubric_support": {},
            "cases": [
                {
                    "case_id": "case_1",
                    "source_pdf": "case_1.pdf",
                    "case_title": "Case 1",
                    "case_subject": "Subject",
                    "prompt": "Prompt",
                    "expected_output": {
                        "political": ["p"],
                        "economic": ["e"],
                        "social": ["s"],
                        "technological": ["t"],
                        "environmental": ["en"],
                        "legal": ["l"],
                        "strategic_implications": ["strategy"],
                    },
                    "gold_claims": [{"category": "Political", "claim": "claim"}],
                    "category_targets": [
                        {
                            "factor": "privacy law",
                            "correct_category": "Legal",
                            "rationale": "law",
                        }
                    ],
                    "avoid_claims": [],
                    "validation_notes": {
                        "source_anchors": [
                            "PESTEL section identifies regulation, innovation, and stakeholder satisfaction.",
                            "Currency inflation supply chain disruption.",
                        ],
                        "reread_justification": "ok",
                    },
                }
            ],
        }
    )


def test_validate_source_anchor_evidence_scores_anchors(monkeypatch):
    monkeypatch.setattr(
        "mascan.eval.source_evidence.extract_pdf_text",
        lambda _path: (
            "The PESTEL section identifies government regulation and product "
            "innovation as drivers of stakeholder satisfaction."
        ),
    )

    report = validate_source_anchor_evidence(_dataset(), match_threshold=0.6)

    assert report.case_count == 1
    assert report.anchor_count == 2
    assert report.matched_anchor_count == 1
    assert report.unmatched_anchor_count == 1
    assert report.extract_error_count == 0
    assert report.evidence[0].matched is True
    assert "innovation" in report.evidence[0].matched_terms
    assert report.evidence[1].matched is False


def test_source_evidence_exports_markdown_and_csv(monkeypatch):
    monkeypatch.setattr(
        "mascan.eval.source_evidence.extract_pdf_text",
        lambda _path: "Regulation and innovation support stakeholder satisfaction.",
    )
    report = validate_source_anchor_evidence(_dataset(), match_threshold=0.3)

    markdown = render_source_evidence_markdown(report)
    rows = source_evidence_csv_rows(report)

    assert "# Gold-Standard Source Anchor Evidence" in markdown
    assert "case_1" in markdown
    assert len(rows) == 2
    assert set(rows[0]) == set(SOURCE_EVIDENCE_CSV_FIELDS)


def test_validate_source_anchor_evidence_reports_extract_errors(monkeypatch):
    def _raise(_path):
        raise ValueError("bad pdf")

    monkeypatch.setattr("mascan.eval.source_evidence.extract_pdf_text", _raise)

    report = validate_source_anchor_evidence(_dataset())

    assert report.extract_error_count == 2
    assert report.evidence[0].error == "bad pdf"
