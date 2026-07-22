from pathlib import Path
from uuid import uuid4

from mascan.eval.gold_analysis import SystemComparison, case_trace_records
from mascan.eval.gold_experiment import (
    JudgedModelResponse,
    MetricPairRecord,
    ModelResponseRecord,
    SystemMetricSummary,
    TokenUsage,
    prompt_sha256,
)
from mascan.eval.gold_judge import (
    GoldJudgeResult,
    gold_judge_prompt_sha256,
    gold_judge_schema_sha256,
)
from mascan.eval.readiness import GoldExperimentManifest, validate_experiment_manifest
from mascan.eval.stats import PairedTestResult


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workspace_tmp() -> Path:
    path = Path("tmp") / "test_readiness" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, payload: object) -> None:
    import json

    _write(path, json.dumps(payload, indent=2, default=str))


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


def _judge(system_id: str) -> GoldJudgeResult:
    return GoldJudgeResult(
        case_id="case_1",
        response_claim_scores=[],
        category_judgments=[
            {
                "factor": "privacy law",
                "expected_category": "Legal",
                "observed_category": "Legal",
                "present": True,
                "correct": True,
                "reasoning": "matched",
            }
        ],
        summary=f"{system_id} ok",
        analytical_depth_score=2.0,
        categorization_accuracy=1.0,
        judge_model="judge",
        judge_prompt_sha256=gold_judge_prompt_sha256(),
        judge_schema_sha256=gold_judge_schema_sha256(),
    )


def _response(system_id: str, model: str) -> ModelResponseRecord:
    if system_id == "mascan":
        generation_config = {
            "runner": "mascan_orchestrator",
            "requested_model": model,
            "effective_default_model": model,
            "agent_models": {"economics": model, "legal": model},
            "prompt_contract": "gold_standard_pestel_v1",
        }
    else:
        generation_config = {
            "runner": "direct_llm",
            "model": model,
            "temperature": 0,
            "max_tokens": 4000,
            "prompt_contract": "gold_standard_pestel_v1",
        }
    return ModelResponseRecord(
        case_id="case_1",
        system_id=system_id,
        model=model,
        prompt_sha256=prompt_sha256("Prompt"),
        generation_config=generation_config,
        response_text=f"{system_id} response",
        token_usage=TokenUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.01
        ),
    )


def test_validate_experiment_manifest_accepts_complete_minimal_run():
    tmp_path = _workspace_tmp()
    _write_json(tmp_path / "gold.json", _gold_payload())
    _write(tmp_path / "case_1.pdf", "%PDF")

    systems = [("mascan", "m1"), ("baseline", "m2")]
    responses = [_response(system_id, model) for system_id, model in systems]
    judged = [
        JudgedModelResponse(response=response, judge=_judge(response.system_id))
        for response in responses
    ]
    _write_json(
        tmp_path / "responses_mascan.json",
        [responses[0].model_dump(mode="json")],
    )
    _write_json(
        tmp_path / "responses_baseline.json",
        [responses[1].model_dump(mode="json")],
    )
    _write_json(tmp_path / "responses_all.json", [r.model_dump(mode="json") for r in responses])
    _write_json(tmp_path / "judged.json", [r.model_dump(mode="json") for r in judged])
    _write_json(tmp_path / "judged_priced.json", [r.model_dump(mode="json") for r in judged])
    _write_json(
        tmp_path / "summary.json",
        [
            SystemMetricSummary(system_id="mascan", n=1).model_dump(mode="json"),
            SystemMetricSummary(system_id="baseline", n=1).model_dump(mode="json"),
        ],
    )
    _write_json(
        tmp_path / "case_trace.json",
        [trace.model_dump(mode="json") for trace in case_trace_records(judged)],
    )
    _write_json(
        tmp_path / "comparison.json",
        SystemComparison(
            treatment_system="mascan",
            control_system="baseline",
            metric="combined_quality",
            paired_cases=1,
            paired_case_ids=["case_1"],
            treatment_scores=[1.0],
            control_scores=[0.9],
            score_pairs=[
                MetricPairRecord(
                    case_id="case_1",
                    treatment_score=1.0,
                    control_score=0.9,
                    difference=0.1,
                )
            ],
            result=PairedTestResult(
                test_name="wilcoxon_signed_rank",
                alternative="two-sided",
                n=1,
                statistic=0.0,
                p_value=1.0,
                mean_difference=0.1,
                normality_alpha=0.05,
            ),
        ).model_dump(mode="json"),
    )
    _write(tmp_path / "report.md", "# Report")

    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {
                "system_id": "mascan",
                "model": "m1",
                "response_file": "responses_mascan.json",
            },
            {
                "system_id": "baseline",
                "model": "m2",
                "response_file": "responses_baseline.json",
            },
        ],
        merged_responses_file="responses_all.json",
        judged_file="judged.json",
        priced_judged_file="judged_priced.json",
        system_summary_file="summary.json",
        case_trace_file="case_trace.json",
        comparisons=[
            {
                "treatment_system": "mascan",
                "control_system": "baseline",
                "metric": "combined_quality",
                "file": "comparison.json",
            }
        ],
        final_report_file="report.md",
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.is_ready is True
    assert report.errors == 0
    artifacts = {fingerprint.artifact for fingerprint in report.fingerprints}
    assert {
        "experiment_manifest",
        "judge_system_prompt",
        "judge_output_schema",
        "gold.json",
        "gold_standard_dataset",
        "responses_mascan.json",
        "responses_baseline.json",
        "responses_all.json",
        "judged.json",
        "judged_priced.json",
        "summary.json",
        "case_trace.json",
        "comparison.json",
        "report.md",
    }.issubset(artifacts)
    assert all(len(fingerprint.sha256) == 64 for fingerprint in report.fingerprints)


def test_validate_experiment_manifest_rejects_stale_comparison_alpha():
    tmp_path = _workspace_tmp()
    _write_json(tmp_path / "gold.json", _gold_payload())
    _write(tmp_path / "case_1.pdf", "%PDF")
    _write_json(
        tmp_path / "comparison.json",
        SystemComparison(
            treatment_system="mascan",
            control_system="baseline",
            metric="combined_quality",
            paired_cases=1,
            paired_case_ids=["case_1"],
            treatment_scores=[1.0],
            control_scores=[0.9],
            score_pairs=[
                MetricPairRecord(
                    case_id="case_1",
                    treatment_score=1.0,
                    control_score=0.9,
                    difference=0.1,
                )
            ],
            result=PairedTestResult(
                test_name="wilcoxon_signed_rank",
                alternative="two-sided",
                n=1,
                statistic=0.0,
                p_value=1.0,
                mean_difference=0.1,
                normality_alpha=0.1,
            ),
        ).model_dump(mode="json"),
    )

    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        comparisons=[
            {
                "treatment_system": "mascan",
                "control_system": "baseline",
                "metric": "combined_quality",
                "normality_alpha": 0.05,
                "file": "comparison.json",
            }
        ],
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.is_ready is False
    assert any("normality_alpha does not match" in issue.message for issue in report.issues)


def test_validate_experiment_manifest_reports_missing_files():
    tmp_path = _workspace_tmp()
    manifest = GoldExperimentManifest(
        gold_standard_file="missing.json",
        expected_case_count=1,
        systems=[],
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.is_ready is False
    assert report.errors == 1
    assert report.issues[0].item == "gold_standard_file"


def test_validate_experiment_manifest_rejects_wrong_prompt_hash():
    tmp_path = _workspace_tmp()
    _write_json(tmp_path / "gold.json", _gold_payload())
    _write(tmp_path / "case_1.pdf", "%PDF")
    response = _response("mascan", "m1").model_copy(
        update={"prompt_sha256": prompt_sha256("different prompt")}
    )
    _write_json(tmp_path / "responses_mascan.json", [response.model_dump(mode="json")])

    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {
                "system_id": "mascan",
                "model": "m1",
                "response_file": "responses_mascan.json",
            }
        ],
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.is_ready is False
    assert any("prompt_sha256 does not match" in issue.message for issue in report.issues)


def test_validate_experiment_manifest_warns_on_pricing_metadata_gap():
    tmp_path = _workspace_tmp()
    _write_json(tmp_path / "gold.json", _gold_payload())
    _write(tmp_path / "case_1.pdf", "%PDF")
    _write_json(
        tmp_path / "pricing.json",
        {
            "prices": [
                {
                    "model": "m1",
                    "prompt_usd_per_1m_tokens": 1.0,
                    "completion_usd_per_1m_tokens": 1.0,
                }
            ]
        },
    )

    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {
                "system_id": "mascan",
                "model": "m1",
                "response_file": "responses_mascan.json",
            }
        ],
        pricing_file="pricing.json",
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.errors == 1
    assert any(issue.item == "response_file:mascan" for issue in report.issues)
    assert any(
        issue.severity == "warning"
        and issue.item == "pricing_file"
        and "source_url" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.severity == "warning"
        and issue.item == "pricing_file"
        and "captured_at" in issue.message
        for issue in report.issues
    )


def test_validate_experiment_manifest_rejects_stale_case_trace_without_audit_counts():
    tmp_path = _workspace_tmp()
    _write_json(tmp_path / "gold.json", _gold_payload())
    _write(tmp_path / "case_1.pdf", "%PDF")
    _write_json(
        tmp_path / "case_trace.json",
        [
            {
                "case_id": "case_1",
                "system_id": "mascan",
                "model": "m1",
                "combined_quality": 0.75,
                "estimated_tokens": False,
            }
        ],
    )

    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {
                "system_id": "mascan",
                "model": "m1",
                "response_file": "responses_mascan.json",
            }
        ],
        case_trace_file="case_trace.json",
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.is_ready is False
    assert any("missing missing_gold_claim_count" in issue.message for issue in report.issues)
    assert any(
        "missing unsupported_or_wrong_claim_count" in issue.message for issue in report.issues
    )


def test_validate_experiment_manifest_rejects_mascan_agent_model_mismatch():
    tmp_path = _workspace_tmp()
    _write_json(tmp_path / "gold.json", _gold_payload())
    _write(tmp_path / "case_1.pdf", "%PDF")
    response = _response("mascan", "gpt-4o-mini")
    response.generation_config["agent_models"] = {
        "economics": "gpt-4o-mini",
        "legal": "gpt-4o",
    }
    _write_json(tmp_path / "responses_mascan.json", [response.model_dump(mode="json")])

    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {
                "system_id": "mascan",
                "model": "gpt-4o-mini",
                "response_file": "responses_mascan.json",
            }
        ],
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.is_ready is False
    assert any("agent_models must all match" in issue.message for issue in report.issues)


def test_validate_experiment_manifest_rejects_direct_llm_generation_config_drift():
    tmp_path = _workspace_tmp()
    _write_json(tmp_path / "gold.json", _gold_payload())
    _write(tmp_path / "case_1.pdf", "%PDF")
    response = _response("zero_shot_same_model", "gpt-4o-mini")
    response.generation_config["temperature"] = 0.7
    response.generation_config["max_tokens"] = 2000
    _write_json(
        tmp_path / "responses_zero.json",
        [response.model_dump(mode="json")],
    )

    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {
                "system_id": "zero_shot_same_model",
                "model": "gpt-4o-mini",
                "response_file": "responses_zero.json",
            }
        ],
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.is_ready is False
    assert any("must use temperature=0" in issue.message for issue in report.issues)
    assert any("must use max_tokens=4000" in issue.message for issue in report.issues)


def test_validate_experiment_manifest_rejects_stale_judge_fingerprints():
    tmp_path = _workspace_tmp()
    _write_json(tmp_path / "gold.json", _gold_payload())
    _write(tmp_path / "case_1.pdf", "%PDF")

    response = _response("mascan", "m1")
    judge = _judge("mascan").model_copy(update={"judge_prompt_sha256": "0" * 64})
    _write_json(
        tmp_path / "judged.json",
        [
            JudgedModelResponse(
                response=response,
                judge=judge,
            ).model_dump(mode="json")
        ],
    )

    manifest = GoldExperimentManifest(
        gold_standard_file="gold.json",
        expected_case_count=1,
        systems=[
            {
                "system_id": "mascan",
                "model": "m1",
                "response_file": "responses_mascan.json",
            }
        ],
        judged_file="judged.json",
    )

    report = validate_experiment_manifest(manifest, base_dir=tmp_path)

    assert report.is_ready is False
    assert any("judge_prompt_sha256 does not match" in issue.message for issue in report.issues)
