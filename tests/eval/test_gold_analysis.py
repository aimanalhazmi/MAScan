import pytest

from mascan.eval.gold_analysis import case_trace_records, compare_systems, summarize_systems
from mascan.eval.gold_experiment import JudgedModelResponse, ModelResponseRecord, TokenUsage
from mascan.eval.gold_judge import (
    GoldJudgeResult,
    gold_judge_prompt_sha256,
    gold_judge_schema_sha256,
)


def _judge(
    case_id: str,
    depth: float,
    accuracy: float,
    *,
    missing: list[str] | None = None,
    unsupported: list[str] | None = None,
) -> GoldJudgeResult:
    return GoldJudgeResult(
        case_id=case_id,
        response_claim_scores=[],
        category_judgments=[],
        missing_gold_claims=missing or [],
        unsupported_or_wrong_claims=unsupported or [],
        summary="ok",
        analytical_depth_score=depth,
        categorization_accuracy=accuracy,
        judge_model="judge",
        judge_prompt_sha256=gold_judge_prompt_sha256(),
        judge_schema_sha256=gold_judge_schema_sha256(),
    )


def _record(case_id: str, system_id: str, depth: float, accuracy: float):
    return JudgedModelResponse(
        response=ModelResponseRecord(case_id=case_id, system_id=system_id, model="m"),
        judge=_judge(case_id, depth, accuracy),
    )


def test_summarize_systems_returns_one_summary_per_system():
    records = [
        _record("a", "mascan", 3.0, 1.0),
        _record("a", "baseline", 2.0, 0.5),
    ]

    summaries = summarize_systems(records)

    assert [summary.system_id for summary in summaries] == ["baseline", "mascan"]


def test_case_trace_records_include_quality_token_and_cost_metrics():
    records = [
        JudgedModelResponse(
            response=ModelResponseRecord(
                case_id="a",
                system_id="mascan",
                model="m",
                prompt_sha256="hash",
                token_usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=100,
                    total_tokens=200,
                    cost_usd=0.01,
                ),
                latency_seconds=1.5,
            ),
            judge=_judge("a", 3.0, 1.0),
        )
    ]
    records[0].judge.missing_gold_claims = ["missing"]
    records[0].judge.unsupported_or_wrong_claims = ["wrong"]

    traces = case_trace_records(records)

    assert len(traces) == 1
    assert traces[0].combined_quality == 1.0
    assert traces[0].quality_per_1k_tokens == 5.0
    assert traces[0].quality_per_usd == 100.0
    assert traces[0].prompt_sha256 == "hash"
    assert traces[0].missing_gold_claim_count == 1
    assert traces[0].unsupported_or_wrong_claim_count == 1


def test_compare_systems_runs_selected_paired_test():
    records = [
        _record("a", "mascan", 3.0, 1.0),
        _record("a", "baseline", 2.0, 0.5),
        _record("b", "mascan", 2.5, 1.0),
        _record("b", "baseline", 2.0, 0.5),
    ]

    comparison = compare_systems(
        records,
        treatment_system="mascan",
        control_system="baseline",
        metric="combined_quality",
        assume_normal=False,
        normality_alpha=0.1,
    )

    assert comparison.paired_cases == 2
    assert comparison.result.test_name == "wilcoxon_signed_rank"
    assert comparison.treatment_scores[0] > comparison.control_scores[0]
    assert comparison.paired_case_ids == ["a", "b"]
    assert comparison.paired_differences == [0.416667, 0.333334]
    assert comparison.mean_difference == 0.375001
    assert comparison.median_difference == 0.375001
    assert comparison.score_pairs[0].case_id == "a"
    assert comparison.result.effect_size_name == "rank_biserial_correlation"
    assert comparison.result.effect_size == 1.0
    assert comparison.result.normality_alpha == 0.1


def test_compare_systems_requires_paired_scores():
    with pytest.raises(ValueError):
        compare_systems(
            [_record("a", "mascan", 3.0, 1.0)],
            treatment_system="mascan",
            control_system="baseline",
            metric="combined_quality",
        )
