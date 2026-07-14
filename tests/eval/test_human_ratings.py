from mascan.eval.gold_experiment import JudgedModelResponse, ModelResponseRecord
from mascan.eval.gold_judge import (
    CategoryTargetJudgment,
    GoldJudgeResult,
    gold_judge_prompt_sha256,
    gold_judge_schema_sha256,
)
from mascan.eval.human_calibration import (
    HumanCalibrationPacket,
    HumanPacketItem,
    HumanPacketOutput,
    HumanAnswerKeyEntry,
    HumanRaterAssignment,
)
from mascan.eval.human_ratings import (
    HumanCategoryRating,
    HumanCategoryRatingTemplate,
    HumanDepthRating,
    HumanDepthRatingTemplate,
    HumanRatingsFile,
    HumanRatingsTemplate,
    build_human_ratings_template,
    compute_human_irr_report,
    discretize_depth_score,
    filter_human_ratings_template,
    human_ratings_from_csv_rows,
    validate_complete_human_ratings,
    validate_human_ratings_template,
)


def _judge(case_id: str, depth: float, correct: bool) -> GoldJudgeResult:
    return GoldJudgeResult(
        case_id=case_id,
        response_claim_scores=[],
        category_judgments=[
            CategoryTargetJudgment(
                factor="privacy law",
                expected_category="Legal",
                observed_category="Legal" if correct else "Technological",
                present=True,
                correct=correct,
                reasoning="test",
            )
        ],
        summary="ok",
        analytical_depth_score=depth,
        categorization_accuracy=1.0 if correct else 0.0,
        judge_model="judge",
        judge_prompt_sha256=gold_judge_prompt_sha256(),
        judge_schema_sha256=gold_judge_schema_sha256(),
    )


def _judged(case_id: str, system_id: str, depth: float, correct: bool):
    return JudgedModelResponse(
        response=ModelResponseRecord(case_id=case_id, system_id=system_id, model="m"),
        judge=_judge(case_id, depth, correct),
    )


def test_discretize_depth_score_maps_llm_average_to_categories():
    assert discretize_depth_score(1.49) == 1
    assert discretize_depth_score(1.5) == 2
    assert discretize_depth_score(2.49) == 2
    assert discretize_depth_score(2.5) == 3


def test_compute_human_irr_report_compares_humans_and_llm():
    answer_key = [
        HumanAnswerKeyEntry(case_id="c1", label="A", system_id="mascan", model="m"),
        HumanAnswerKeyEntry(case_id="c1", label="B", system_id="baseline", model="b"),
    ]
    judged = [
        _judged("c1", "mascan", 3.0, True),
        _judged("c1", "baseline", 2.0, False),
    ]
    ratings = HumanRatingsFile(
        depth_ratings=[
            HumanDepthRating(
                rater_id="r1", case_id="c1", label="A", analytical_depth_score=3
            ),
            HumanDepthRating(
                rater_id="r1", case_id="c1", label="B", analytical_depth_score=2
            ),
            HumanDepthRating(
                rater_id="r2", case_id="c1", label="A", analytical_depth_score=3
            ),
            HumanDepthRating(
                rater_id="r2", case_id="c1", label="B", analytical_depth_score=1
            ),
        ],
        category_ratings=[
            HumanCategoryRating(
                rater_id="r1",
                case_id="c1",
                label="A",
                factor="privacy law",
                correct=True,
            ),
            HumanCategoryRating(
                rater_id="r1",
                case_id="c1",
                label="B",
                factor="privacy law",
                correct=False,
            ),
            HumanCategoryRating(
                rater_id="r2",
                case_id="c1",
                label="A",
                factor="privacy law",
                correct=True,
            ),
            HumanCategoryRating(
                rater_id="r2",
                case_id="c1",
                label="B",
                factor="privacy law",
                correct=False,
            ),
        ],
    )

    report = compute_human_irr_report(ratings, judged, answer_key)

    assert report.depth_fleiss is not None
    assert report.depth_fleiss.n_items == 2
    assert report.depth_fleiss_with_llm is not None
    assert report.depth_fleiss_with_llm.n_items == 2
    assert report.depth_cohen_pooled is not None
    assert report.depth_cohen_pooled.n_items == 4
    assert len(report.depth_cohen_by_rater) == 2
    assert report.depth_weighted_cohen_pooled is not None
    assert report.depth_weighted_cohen_pooled.n_items == 4
    assert len(report.depth_weighted_cohen_by_rater) == 2
    assert report.category_fleiss is not None
    assert report.category_fleiss.kappa == 1.0
    assert report.category_fleiss_with_llm is not None
    assert report.category_fleiss_with_llm.kappa == 1.0
    assert report.category_cohen_pooled is not None
    assert report.category_cohen_pooled.kappa == 1.0
    assert len(report.category_cohen_by_rater) == 2


def _packet() -> HumanCalibrationPacket:
    return HumanCalibrationPacket(
        seed=1,
        selected_case_ids=["c1"],
        cases_per_rater=1,
        rater_assignments=[HumanRaterAssignment(rater_id="r1", case_ids=["c1"])],
        instructions="rate",
        rating_scale={"1": "surface", "2": "impact", "3": "strategy"},
        items=[
            HumanPacketItem(
                case_id="c1",
                case_title="case",
                prompt="prompt",
                expected_output={"political": ["x"]},
                category_targets=[
                    {
                        "factor": "privacy law",
                        "correct_category": "Legal",
                        "rationale": "law",
                    }
                ],
                outputs=[
                    HumanPacketOutput(label="A", response_text="a"),
                    HumanPacketOutput(label="B", response_text="b"),
                ],
            )
        ],
    )


def test_build_human_ratings_template_has_rows_for_assigned_raters_only():
    template = build_human_ratings_template(_packet(), rater_ids=["r1", "r2"])

    assert len(template.depth_ratings) == 2
    assert len(template.category_ratings) == 2
    assert {rating.rater_id for rating in template.depth_ratings} == {"r1"}
    assert template.category_ratings[0].correct_category == "Legal"
    assert template.category_ratings[0].rationale == "law"
    assert template.category_ratings[0].correct is None


def test_validate_human_ratings_template_rejects_stale_or_prefilled_rows():
    template = HumanRatingsTemplate(
        depth_ratings=[
            HumanDepthRatingTemplate(
                rater_id="r1",
                case_id="c1",
                label="A",
                analytical_depth_score=2,
            )
        ],
        category_ratings=[
            HumanCategoryRatingTemplate(
                rater_id="r1",
                case_id="c1",
                label="A",
                factor="privacy law",
                correct_category=None,
                rationale=None,
            )
        ],
    )

    report = validate_human_ratings_template(template, _packet(), rater_ids=["r1"])

    assert report.is_valid is False
    assert "r1|c1|B" in report.missing_depth_rows
    assert "r1|c1|A" in report.prefilled_depth_rows
    assert report.category_context_mismatches


def test_filter_human_ratings_template_returns_one_rater():
    template = build_human_ratings_template(_packet(), rater_ids=["r1", "r2"])

    r1_template = filter_human_ratings_template(template, rater_id="r1")

    assert {rating.rater_id for rating in r1_template.depth_ratings} == {"r1"}
    assert {rating.rater_id for rating in r1_template.category_ratings} == {"r1"}
    assert len(r1_template.depth_ratings) == 2
    assert len(r1_template.category_ratings) == 2


def test_human_ratings_from_csv_rows_parses_filled_exports():
    ratings = human_ratings_from_csv_rows(
        [
            {
                "metric": "analytical_depth",
                "rater_id": "r1",
                "case_id": "c1",
                "label": "A",
                "factor": "",
                "analytical_depth_score": "3",
                "correct": "",
            },
            {
                "metric": "categorization_accuracy",
                "rater_id": "r1",
                "case_id": "c1",
                "label": "A",
                "factor": "privacy law",
                "analytical_depth_score": "",
                "correct": "yes",
            },
        ]
    )

    assert ratings.depth_ratings[0].analytical_depth_score == 3
    assert ratings.category_ratings[0].correct is True


def test_validate_complete_human_ratings_accepts_full_file():
    ratings = HumanRatingsFile(
        depth_ratings=[
            HumanDepthRating(
                rater_id="r1", case_id="c1", label=label, analytical_depth_score=2
            )
            for label in ["A", "B"]
        ],
        category_ratings=[
            HumanCategoryRating(
                rater_id="r1",
                case_id="c1",
                label=label,
                factor="privacy law",
                correct=True,
            )
            for label in ["A", "B"]
        ],
    )

    report = validate_complete_human_ratings(
        ratings, _packet(), rater_ids=["r1", "r2"]
    )

    assert report.is_complete is True
    assert report.expected_depth_count == 2
    assert report.expected_category_count == 2


def test_compute_human_irr_report_omits_fleiss_for_distributed_ratings():
    answer_key = [
        HumanAnswerKeyEntry(case_id="c1", label="A", system_id="mascan", model="m"),
        HumanAnswerKeyEntry(case_id="c1", label="B", system_id="baseline", model="b"),
    ]
    judged = [
        _judged("c1", "mascan", 3.0, True),
        _judged("c1", "baseline", 2.0, False),
    ]
    ratings = HumanRatingsFile(
        depth_ratings=[
            HumanDepthRating(
                rater_id="r1", case_id="c1", label="A", analytical_depth_score=3
            ),
            HumanDepthRating(
                rater_id="r1", case_id="c1", label="B", analytical_depth_score=2
            ),
        ],
        category_ratings=[
            HumanCategoryRating(
                rater_id="r1",
                case_id="c1",
                label="A",
                factor="privacy law",
                correct=True,
            ),
            HumanCategoryRating(
                rater_id="r1",
                case_id="c1",
                label="B",
                factor="privacy law",
                correct=False,
            ),
        ],
    )

    report = compute_human_irr_report(ratings, judged, answer_key)

    assert report.depth_fleiss is None
    assert report.depth_fleiss_with_llm is None
    assert report.depth_cohen_pooled is not None
    assert report.category_fleiss is None


def test_validate_complete_human_ratings_reports_missing_and_duplicate():
    ratings = HumanRatingsFile(
        depth_ratings=[
            HumanDepthRating(
                rater_id="r1", case_id="c1", label="A", analytical_depth_score=2
            ),
            HumanDepthRating(
                rater_id="r1", case_id="c1", label="A", analytical_depth_score=3
            ),
        ],
        category_ratings=[],
    )

    report = validate_complete_human_ratings(
        ratings, _packet(), rater_ids=["r1", "r2"]
    )

    assert report.is_complete is False
    assert "r1|c1|A" in report.duplicate_depth_ratings
    assert report.missing_depth_ratings
    assert report.missing_category_ratings
