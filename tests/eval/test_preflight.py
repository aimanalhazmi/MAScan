import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from mascan.eval.costing import ModelPricing, PricingTable
from mascan.eval.preflight import (
    GoldPreflightReport,
    PreflightIssue,
    render_gold_preflight_markdown,
    run_gold_preflight,
)
from mascan.eval.readiness import GoldExperimentManifest


def _workspace_tmp() -> Path:
    path = Path("tmp") / "test_preflight" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _gold_payload() -> dict[str, object]:
    return {
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
                    "source_anchors": ["anchor"],
                    "reread_justification": "ok",
                },
            }
        ],
    }


def _manifest() -> GoldExperimentManifest:
    return GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {
                "system_id": "mascan",
                "model": "m1",
                "response_file": "responses_mascan.json",
            },
            {
                "system_id": "zero_shot_same_model",
                "model": "m1",
                "response_file": "responses_zero.json",
            },
            {
                "system_id": "frontier_model",
                "model": "m2",
                "response_file": "responses_frontier.json",
            },
        ],
        pricing_file="pricing.json",
        human_calibration={
            "packet_file": "packet.json",
            "answer_key_file": "answer_key.json",
            "ratings_file": "human_ratings.json",
            "rater_ids": ["r1"],
            "cases_per_rater": 1,
            "expected_case_count": 1,
        },
    )


def _write_inputs(base: Path) -> None:
    (base / "gold.json").write_text(json.dumps(_gold_payload()), encoding="utf-8")
    (base / "case_1.pdf").write_text("%PDF", encoding="utf-8")
    pricing = PricingTable(
        prices=[
            ModelPricing(
                model="m1",
                prompt_usd_per_1m_tokens=1,
                completion_usd_per_1m_tokens=2,
            ),
            ModelPricing(
                model="m2",
                prompt_usd_per_1m_tokens=3,
                completion_usd_per_1m_tokens=4,
            ),
        ]
    )
    (base / "pricing.json").write_text(pricing.model_dump_json(), encoding="utf-8")


def test_pre_human_preflight_accepts_required_inputs(monkeypatch):
    base = _workspace_tmp()
    _write_inputs(base)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "mascan.eval.preflight.sys.version_info",
        SimpleNamespace(major=3, minor=12, micro=0),
    )
    monkeypatch.setattr(
        "mascan.eval.preflight.importlib.util.find_spec",
        lambda _name: object(),
    )

    report = run_gold_preflight(_manifest(), base_dir=base)

    assert report.is_ready is True
    assert report.errors == 0
    assert {
        issue.message
        for issue in report.issues
        if issue.item == "pricing_file"
    } == {
        "Pricing table is missing citation metadata field: source_url",
        "Pricing table is missing citation metadata field: captured_at",
    }


def test_pre_human_preflight_reports_missing_key_and_pricing(monkeypatch):
    base = _workspace_tmp()
    (base / "gold.json").write_text(json.dumps(_gold_payload()), encoding="utf-8")
    (base / "case_1.pdf").write_text("%PDF", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "mascan.eval.preflight.sys.version_info",
        SimpleNamespace(major=3, minor=12, micro=0),
    )
    monkeypatch.setattr(
        "mascan.eval.preflight.importlib.util.find_spec",
        lambda _name: object(),
    )

    report = run_gold_preflight(_manifest(), base_dir=base)

    assert report.is_ready is False
    assert any(issue.item == "env:OPENAI_API_KEY" for issue in report.issues)
    assert any(issue.item == "pricing_file" for issue in report.issues)


def test_pre_human_preflight_blocks_wrong_python(monkeypatch):
    base = _workspace_tmp()
    _write_inputs(base)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "mascan.eval.preflight.sys.version_info",
        SimpleNamespace(major=3, minor=14, micro=3),
    )
    monkeypatch.setattr(
        "mascan.eval.preflight.importlib.util.find_spec",
        lambda _name: object(),
    )

    report = run_gold_preflight(_manifest(), base_dir=base)

    assert report.is_ready is False
    assert any(
        issue.severity == "error" and issue.item == "python_version"
        for issue in report.issues
    )


def test_render_preflight_markdown_includes_action_checklist():
    report = GoldPreflightReport(
        is_ready=False,
        phase="pre_human",
        errors=1,
        warnings=1,
        issues=[
            PreflightIssue(
                severity="error",
                item="python_version",
                message="Current runtime is 3.14.3.",
            ),
            PreflightIssue(
                severity="warning",
                item="pricing_file",
                message="Pricing table is missing citation metadata field: source_url",
            ),
        ],
    )

    rendered = render_gold_preflight_markdown(report)

    assert "# Gold Experiment Preflight Report" in rendered
    assert "- Status: blocked" in rendered
    assert "`python_version`" in rendered
    assert "activated Python 3.12" in rendered
    assert "`pricing_file`" in rendered
    assert "Add the source URL" in rendered
