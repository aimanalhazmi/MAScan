import pytest

from mascan.eval import gold_judge
from mascan.eval.gold_judge import (
    CategoryTargetJudgment,
    ResponseClaimScore,
    _LLMGoldJudgeOutput,
)
from mascan.eval.gold_standard import load_gold_standard


def test_compute_analytical_depth_is_mechanical_average():
    scores = [
        ResponseClaimScore(response_claim="A", score=1, reasoning="surface"),
        ResponseClaimScore(response_claim="B", score=2, reasoning="impact"),
        ResponseClaimScore(response_claim="C", score=3, reasoning="strategy"),
    ]

    assert gold_judge.compute_analytical_depth(scores) == 2.0
    assert gold_judge.compute_analytical_depth([]) == 1.0


def test_compute_categorization_accuracy_counts_missing_as_wrong():
    judgments = [
        CategoryTargetJudgment(
            factor="privacy law",
            expected_category="Legal",
            observed_category="Legal",
            present=True,
            correct=True,
            reasoning="ok",
        ),
        CategoryTargetJudgment(
            factor="AI",
            expected_category="Technological",
            observed_category="Social",
            present=True,
            correct=False,
            reasoning="wrong bucket",
        ),
        CategoryTargetJudgment(
            factor="inflation",
            expected_category="Economic",
            observed_category=None,
            present=False,
            correct=False,
            reasoning="missing",
        ),
    ]

    assert gold_judge.compute_categorization_accuracy(judgments) == 0.3333
    assert gold_judge.compute_categorization_accuracy(judgments, present_only=True) == 0.5


def test_build_gold_judge_prompt_includes_strict_targets():
    case = load_gold_standard().by_id("2007_1_SHELL")

    prompt = gold_judge.build_gold_judge_user_prompt(case, "Political: policy matters")

    assert "2007_1_SHELL" in prompt
    assert "gold_claims" in prompt
    assert "category_targets" in prompt
    assert "Political: policy matters" in prompt


def test_gold_judge_output_schema_exposes_required_fields():
    schema = gold_judge.gold_judge_output_schema()

    assert {
        "response_claim_scores",
        "category_judgments",
        "missing_gold_claims",
        "unsupported_or_wrong_claims",
        "summary",
    }.issubset(schema["properties"])


def test_gold_judge_fingerprints_are_stable_sha256_values():
    prompt_hash = gold_judge.gold_judge_prompt_sha256()
    schema_hash = gold_judge.gold_judge_schema_sha256()

    assert len(prompt_hash) == 64
    assert len(schema_hash) == 64
    assert prompt_hash != schema_hash


def test_validate_category_judgment_alignment_rejects_missing_targets():
    case = load_gold_standard().by_id("2007_1_SHELL")

    with pytest.raises(ValueError):
        gold_judge.validate_category_judgment_alignment(case, [])


def test_judge_gold_response_constructs_metrics(mocker):
    case = load_gold_standard().by_id("2007_1_SHELL")
    category_judgments = []
    for index, target in enumerate(case.category_targets):
        present = index < 3
        category_judgments.append(
            CategoryTargetJudgment(
                factor=target.factor,
                expected_category=target.correct_category,
                observed_category=target.correct_category if present else None,
                present=present,
                correct=present,
                reasoning="matched" if present else "missing",
            )
        )
    payload = _LLMGoldJudgeOutput(
        response_claim_scores=[
            ResponseClaimScore(
                response_claim="EU rules raise compliance cost",
                category="Legal",
                linked_gold_claims=["EU penalties"],
                score=2,
                reasoning="impact but no strategy",
            ),
            ResponseClaimScore(
                response_claim="Cleaner processes reduce cost and differentiate",
                category="Technological",
                linked_gold_claims=["Cleaner processes"],
                score=3,
                reasoning="strategy",
            ),
        ],
        category_judgments=category_judgments,
        summary="mixed",
    )
    structured = mocker.Mock()
    structured.invoke.return_value = payload
    model = mocker.Mock()
    model.with_structured_output.return_value = structured
    mocker.patch.object(gold_judge, "get_chat_model", return_value=model)

    result = gold_judge.judge_gold_response(case, "answer", model="gpt-4o")

    assert result.case_id == "2007_1_SHELL"
    assert result.analytical_depth_score == 2.5
    assert result.categorization_accuracy == 0.5
    assert result.categorization_accuracy_present_only == 1.0
    assert result.response_claim_count == 2
    assert result.category_targets_evaluated == 6
    assert result.category_targets_present == 3
    assert result.category_targets_missing == 3
    assert result.category_targets_correct == 3
    assert result.judge_model == "gpt-4o"
    assert result.judge_prompt_sha256 == gold_judge.gold_judge_prompt_sha256()
    assert result.judge_schema_sha256 == gold_judge.gold_judge_schema_sha256()
