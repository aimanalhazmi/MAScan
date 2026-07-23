"""Test-only helpers for building complete gold-standard experiment fixtures."""

import json
from pathlib import Path

from mascan.eval.costing import ModelPricing, PricingTable, apply_pricing_to_judged
from mascan.eval.gold_analysis import case_trace_records, compare_systems, summarize_systems
from mascan.eval.gold_experiment import (
    CONTROL_GROUP_SYSTEMS,
    JudgedModelResponse,
    ModelResponseRecord,
    estimate_token_usage,
    prompt_sha256,
)
from mascan.eval.gold_judge import (
    CategoryTargetJudgment,
    GoldJudgeResult,
    ResponseClaimScore,
    gold_judge_prompt_sha256,
    gold_judge_schema_sha256,
)
from mascan.eval.gold_report import render_gold_experiment_report
from mascan.eval.gold_standard import GoldStandardCase, load_gold_standard
from mascan.eval.readiness import GoldExperimentManifest, validate_experiment_manifest

SYSTEM_MODELS = {
    "mascan": "gpt-4o-mini",
    "zero_shot_same_model": "gpt-4o-mini",
    "frontier_model": "gpt-4o",
}


def build_complete_gold_run_fixture(
    out_dir: str | Path,
    *,
    gold_standard_path: str | Path = "eval_papers/gold_standard_cases.json",
    seed: int = 20260712,
) -> Path:
    """Write a structurally complete gold experiment under out_dir for tests."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_gold_standard(gold_standard_path)
    systems = list(CONTROL_GROUP_SYSTEMS)

    response_records: list[ModelResponseRecord] = []
    for system_id in systems:
        records = [
            _response(case, system_id=system_id, model=SYSTEM_MODELS[system_id])
            for case in dataset.cases
        ]
        response_records.extend(records)
        _write_json(
            output / f"responses_{system_id}.json",
            [record.model_dump(mode="json") for record in records],
        )

    response_records = sorted(
        response_records, key=lambda record: (record.case_id, record.system_id)
    )
    _write_json(
        output / "responses_all.json",
        [record.model_dump(mode="json") for record in response_records],
    )

    judged_records = [
        JudgedModelResponse(
            response=record,
            judge=_judge(dataset.by_id(record.case_id), record.system_id),
        )
        for record in response_records
    ]
    _write_json(
        output / "judged_all.json",
        [record.model_dump(mode="json") for record in judged_records],
    )

    pricing = PricingTable(
        source_url="test://gold-fixture-pricing",
        captured_at="2026-07-12",
        notes="Test fixture pricing only.",
        prices=[
            ModelPricing(
                model="gpt-4o-mini",
                prompt_usd_per_1m_tokens=0.15,
                completion_usd_per_1m_tokens=0.60,
            ),
            ModelPricing(
                model="gpt-4o",
                prompt_usd_per_1m_tokens=2.50,
                completion_usd_per_1m_tokens=10.00,
            ),
        ],
    )
    _write_json(output / "model_pricing.json", pricing.model_dump(mode="json"))
    priced_judged = apply_pricing_to_judged(judged_records, pricing)
    _write_json(
        output / "judged_all_priced.json",
        [record.model_dump(mode="json") for record in priced_judged],
    )

    summaries = summarize_systems(priced_judged)
    _write_json(
        output / "system_summary.json",
        [summary.model_dump(mode="json") for summary in summaries],
    )
    traces = case_trace_records(priced_judged)
    _write_json(
        output / "case_trace.json",
        [trace.model_dump(mode="json") for trace in traces],
    )

    comparisons = [
        compare_systems(
            priced_judged,
            treatment_system="mascan",
            control_system="zero_shot_same_model",
            metric="combined_quality",
            assume_normal=False,
        ),
        compare_systems(
            priced_judged,
            treatment_system="mascan",
            control_system="frontier_model",
            metric="combined_quality",
            assume_normal=False,
        ),
    ]
    comparison_files = [
        output / "mascan_vs_zero_shot.json",
        output / "mascan_vs_frontier.json",
    ]
    for comparison_file, comparison in zip(comparison_files, comparisons, strict=True):
        _write_json(comparison_file, comparison.model_dump(mode="json"))

    report_text = render_gold_experiment_report(
        summaries,
        comparisons=comparisons,
    )
    (output / "gold_experiment_report.md").write_text(report_text, encoding="utf-8")

    manifest = _manifest(output, gold_standard_path)
    manifest_path = output / "gold_experiment_manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    readiness = validate_experiment_manifest(manifest, base_dir=".")
    _write_json(output / "readiness_report.json", readiness.model_dump(mode="json"))
    return manifest_path


def _response(case: GoldStandardCase, *, system_id: str, model: str) -> ModelResponseRecord:
    first_political = case.expected_output.political[0] if case.expected_output.political else ""
    response_text = (
        f"Fixture {system_id} response for {case.case_id}.\n\n"
        f"Political: {first_political}\n"
        "Economic: External pressures affect operations and strategy.\n"
        "Strategic implications: Test fixture content."
    )
    return ModelResponseRecord(
        case_id=case.case_id,
        system_id=system_id,
        model=model,
        prompt_sha256=prompt_sha256(case.prompt),
        generation_config=_generation_config(system_id, model),
        response_text=response_text,
        token_usage=estimate_token_usage(case.prompt, response_text),
        latency_seconds=0.01,
    )


def _generation_config(system_id: str, model: str) -> dict[str, object]:
    if system_id == "mascan":
        return {
            "prompt_contract": "gold_standard_pestel_v1",
            "runner": "mascan_orchestrator",
            "requested_model": model,
            "effective_default_model": model,
            "agent_models": {
                "economics": model,
                "environmental": model,
                "legal": model,
                "political": model,
                "social": model,
            },
        }
    return {
        "prompt_contract": "gold_standard_pestel_v1",
        "runner": "direct_llm",
        "model": model,
        "temperature": 0,
        "max_tokens": 4000,
    }


def _judge(case: GoldStandardCase, system_id: str) -> GoldJudgeResult:
    depth = {"zero_shot_same_model": 2, "mascan": 3, "frontier_model": 3}[system_id]
    correct_limit = {
        "zero_shot_same_model": max(1, len(case.category_targets) // 2),
        "mascan": len(case.category_targets),
        "frontier_model": len(case.category_targets),
    }[system_id]
    category_judgments = []
    for index, target in enumerate(case.category_targets):
        correct = index < correct_limit
        category_judgments.append(
            CategoryTargetJudgment(
                factor=target.factor,
                expected_category=target.correct_category,
                observed_category=target.correct_category if correct else "Wrong",
                present=True,
                correct=correct,
                reasoning="Fixture judgment.",
            )
        )
    accuracy = sum(1 for judgment in category_judgments if judgment.correct) / len(
        category_judgments
    )
    return GoldJudgeResult(
        case_id=case.case_id,
        response_claim_scores=[
            ResponseClaimScore(
                response_claim=f"Fixture {system_id} causal claim.",
                category="Political",
                linked_gold_claims=[case.gold_claims[0].claim] if case.gold_claims else [],
                score=depth,
                reasoning="Fixture score.",
            )
        ],
        category_judgments=category_judgments,
        missing_gold_claims=[],
        unsupported_or_wrong_claims=[],
        summary="Fixture judge output.",
        analytical_depth_score=float(depth),
        categorization_accuracy=round(accuracy, 4),
        categorization_accuracy_present_only=round(accuracy, 4),
        judge_model="fixture-judge",
        judge_prompt_sha256=gold_judge_prompt_sha256(),
        judge_schema_sha256=gold_judge_schema_sha256(),
    )


def _manifest(out_dir: Path, gold_standard_path: str | Path) -> GoldExperimentManifest:
    def path(name: str) -> str:
        return str((out_dir / name).as_posix())

    return GoldExperimentManifest(
        gold_standard_file=str(Path(gold_standard_path).as_posix()),
        expected_case_count=25,
        systems=[
            {
                "system_id": system_id,
                "model": SYSTEM_MODELS[system_id],
                "response_file": path(f"responses_{system_id}.json"),
            }
            for system_id in CONTROL_GROUP_SYSTEMS
        ],
        merged_responses_file=path("responses_all.json"),
        judged_file=path("judged_all.json"),
        priced_judged_file=path("judged_all_priced.json"),
        pricing_file=path("model_pricing.json"),
        system_summary_file=path("system_summary.json"),
        case_trace_file=path("case_trace.json"),
        comparisons=[
            {
                "treatment_system": "mascan",
                "control_system": "zero_shot_same_model",
                "metric": "combined_quality",
                "file": path("mascan_vs_zero_shot.json"),
            },
            {
                "treatment_system": "mascan",
                "control_system": "frontier_model",
                "metric": "combined_quality",
                "file": path("mascan_vs_frontier.json"),
            },
        ],
        final_report_file=path("gold_experiment_report.md"),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
