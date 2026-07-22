from mascan.eval.gold_experiment import (
    JudgedModelResponse,
    ModelResponseRecord,
    TokenUsage,
    combined_quality_score,
    estimate_token_usage,
    metric_pair_records,
    metric_pairs,
    prompt_sha256,
    summarize_system,
)
from mascan.eval.gold_judge import (
    GoldJudgeResult,
    gold_judge_prompt_sha256,
    gold_judge_schema_sha256,
)


def _judge(
    depth: float,
    accuracy: float,
    *,
    missing: list[str] | None = None,
    unsupported: list[str] | None = None,
) -> GoldJudgeResult:
    return GoldJudgeResult(
        case_id="case",
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


def test_estimate_token_usage_is_deterministic():
    usage = estimate_token_usage("abcd", "abcdefgh")

    assert usage.prompt_tokens == 1
    assert usage.completion_tokens == 2
    assert usage.total_tokens == 3
    assert usage.estimated is True


def test_prompt_sha256_is_stable_for_same_prompt():
    assert prompt_sha256("same prompt") == prompt_sha256("same prompt")
    assert prompt_sha256("same prompt") != prompt_sha256("different prompt")


def test_combined_quality_normalizes_depth_and_accuracy():
    assert combined_quality_score(_judge(3.0, 1.0)) == 1.0
    assert combined_quality_score(_judge(1.5, 0.5)) == 0.5


def test_metric_pairs_align_by_case_id():
    records = [
        JudgedModelResponse(
            response=ModelResponseRecord(case_id="a", system_id="mascan", model="m"),
            judge=_judge(3.0, 1.0),
        ),
        JudgedModelResponse(
            response=ModelResponseRecord(case_id="a", system_id="baseline", model="b"),
            judge=_judge(2.0, 0.5),
        ),
        JudgedModelResponse(
            response=ModelResponseRecord(case_id="b", system_id="mascan", model="m"),
            judge=_judge(1.0, 0.0),
        ),
    ]

    treatment, control = metric_pairs(
        records,
        treatment_system="mascan",
        control_system="baseline",
        metric="analytical_depth",
    )

    assert treatment == [3.0]
    assert control == [2.0]

    pair_records = metric_pair_records(
        records,
        treatment_system="mascan",
        control_system="baseline",
        metric="analytical_depth",
    )
    assert pair_records[0].case_id == "a"
    assert pair_records[0].difference == 1.0


def test_summarize_system_includes_token_efficiency():
    records = [
        JudgedModelResponse(
            response=ModelResponseRecord(
                case_id="a",
                system_id="mascan",
                model="m",
                token_usage=TokenUsage(total_tokens=100, cost_usd=0.01),
            ),
            judge=_judge(3.0, 1.0, missing=["claim a"], unsupported=["bad a"]),
        ),
        JudgedModelResponse(
            response=ModelResponseRecord(
                case_id="b",
                system_id="mascan",
                model="m",
                token_usage=TokenUsage(total_tokens=100, cost_usd=0.01),
            ),
            judge=_judge(1.5, 0.5, missing=["claim b", "claim c"]),
        ),
    ]

    summary = summarize_system("mascan", records)

    assert summary.n == 2
    assert summary.mean_analytical_depth == 2.25
    assert summary.mean_categorization_accuracy == 0.75
    assert summary.total_quality_points == 1.5
    assert summary.total_missing_gold_claims == 3
    assert summary.mean_missing_gold_claims == 1.5
    assert summary.total_unsupported_or_wrong_claims == 1
    assert summary.mean_unsupported_or_wrong_claims == 0.5
    assert summary.total_tokens == 200
    assert summary.total_cost_usd == 0.02
    assert summary.quality_per_1k_tokens == 7.5
    assert summary.quality_per_usd == 75.0
