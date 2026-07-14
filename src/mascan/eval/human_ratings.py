"""Human-rating contracts and IRR analysis for the calibration packet."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from mascan.eval.gold_experiment import JudgedModelResponse
from mascan.eval.human_calibration import (
    HumanAnswerKeyEntry,
    HumanCalibrationPacket,
    assigned_rater_id,
)
from mascan.eval.irr import cohen_kappa, fleiss_kappa, weighted_cohen_kappa


class HumanDepthRating(BaseModel):
    rater_id: str
    case_id: str
    label: str
    analytical_depth_score: int = Field(..., ge=1, le=3)


class HumanCategoryRating(BaseModel):
    rater_id: str
    case_id: str
    label: str
    factor: str
    correct: bool


class HumanRatingsFile(BaseModel):
    depth_ratings: list[HumanDepthRating] = Field(default_factory=list)
    category_ratings: list[HumanCategoryRating] = Field(default_factory=list)


class HumanDepthRatingTemplate(BaseModel):
    rater_id: str
    case_id: str
    label: str
    analytical_depth_score: int | None = Field(default=None, ge=1, le=3)


class HumanCategoryRatingTemplate(BaseModel):
    rater_id: str
    case_id: str
    label: str
    factor: str
    correct_category: str | None = None
    rationale: str | None = None
    correct: bool | None = None


class HumanRatingsTemplate(BaseModel):
    depth_ratings: list[HumanDepthRatingTemplate] = Field(default_factory=list)
    category_ratings: list[HumanCategoryRatingTemplate] = Field(default_factory=list)


class HumanRatingsValidationReport(BaseModel):
    is_complete: bool
    expected_depth_count: int
    observed_depth_count: int
    expected_category_count: int
    observed_category_count: int
    missing_depth_ratings: list[str] = Field(default_factory=list)
    missing_category_ratings: list[str] = Field(default_factory=list)
    duplicate_depth_ratings: list[str] = Field(default_factory=list)
    duplicate_category_ratings: list[str] = Field(default_factory=list)
    unexpected_depth_ratings: list[str] = Field(default_factory=list)
    unexpected_category_ratings: list[str] = Field(default_factory=list)


class HumanRatingsTemplateValidationReport(BaseModel):
    is_valid: bool
    expected_depth_count: int
    observed_depth_count: int
    expected_category_count: int
    observed_category_count: int
    missing_depth_rows: list[str] = Field(default_factory=list)
    missing_category_rows: list[str] = Field(default_factory=list)
    duplicate_depth_rows: list[str] = Field(default_factory=list)
    duplicate_category_rows: list[str] = Field(default_factory=list)
    unexpected_depth_rows: list[str] = Field(default_factory=list)
    unexpected_category_rows: list[str] = Field(default_factory=list)
    category_context_mismatches: list[str] = Field(default_factory=list)
    prefilled_depth_rows: list[str] = Field(default_factory=list)
    prefilled_category_rows: list[str] = Field(default_factory=list)


class KappaSummary(BaseModel):
    metric: str
    comparison: str
    n_items: int
    kappa: float | None


class HumanIrrReport(BaseModel):
    depth_fleiss: KappaSummary | None = None
    depth_fleiss_with_llm: KappaSummary | None = None
    depth_cohen_pooled: KappaSummary | None = None
    depth_cohen_by_rater: list[KappaSummary] = Field(default_factory=list)
    depth_weighted_cohen_pooled: KappaSummary | None = None
    depth_weighted_cohen_by_rater: list[KappaSummary] = Field(default_factory=list)
    category_fleiss: KappaSummary | None = None
    category_fleiss_with_llm: KappaSummary | None = None
    category_cohen_pooled: KappaSummary | None = None
    category_cohen_by_rater: list[KappaSummary] = Field(default_factory=list)


def discretize_depth_score(score: float) -> int:
    """Map the LLM judge's average 1-3 score back to a kappa category."""
    if score < 1.5:
        return 1
    if score < 2.5:
        return 2
    return 3


def build_human_ratings_template(
    packet: HumanCalibrationPacket,
    *,
    rater_ids: Sequence[str] | None = None,
) -> HumanRatingsTemplate:
    """Build a fillable JSON template for the packet's assigned raters."""
    depth_ratings: list[HumanDepthRatingTemplate] = []
    category_ratings: list[HumanCategoryRatingTemplate] = []
    for item in packet.items:
        rater_id = assigned_rater_id(packet, item.case_id)
        if rater_ids is not None and rater_id not in rater_ids:
            continue
        for output in item.outputs:
            depth_ratings.append(
                HumanDepthRatingTemplate(
                    rater_id=rater_id,
                    case_id=item.case_id,
                    label=output.label,
                )
            )
            for target in item.category_targets:
                category_ratings.append(
                    HumanCategoryRatingTemplate(
                        rater_id=rater_id,
                        case_id=item.case_id,
                        label=output.label,
                        factor=target["factor"],
                        correct_category=target.get("correct_category"),
                        rationale=target.get("rationale"),
                    )
                )
    return HumanRatingsTemplate(
        depth_ratings=depth_ratings,
        category_ratings=category_ratings,
    )


def filter_human_ratings_template(
    template: HumanRatingsTemplate,
    *,
    rater_id: str,
) -> HumanRatingsTemplate:
    """Return only the rows assigned to one human rater."""
    return HumanRatingsTemplate(
        depth_ratings=[
            rating for rating in template.depth_ratings if rating.rater_id == rater_id
        ],
        category_ratings=[
            rating for rating in template.category_ratings if rating.rater_id == rater_id
        ],
    )


def human_ratings_from_csv_rows(
    rows: Iterable[Mapping[str, str]],
) -> HumanRatingsFile:
    """Parse filled ratings CSV rows exported by the calibration workflow."""
    depth_ratings: list[HumanDepthRating] = []
    category_ratings: list[HumanCategoryRating] = []
    for row_number, row in enumerate(rows, start=2):
        metric = _required_csv_value(row, "metric", row_number)
        if metric == "analytical_depth":
            depth_ratings.append(
                HumanDepthRating(
                    rater_id=_required_csv_value(row, "rater_id", row_number),
                    case_id=_required_csv_value(row, "case_id", row_number),
                    label=_required_csv_value(row, "label", row_number),
                    analytical_depth_score=_parse_depth_score(row, row_number),
                )
            )
        elif metric == "categorization_accuracy":
            category_ratings.append(
                HumanCategoryRating(
                    rater_id=_required_csv_value(row, "rater_id", row_number),
                    case_id=_required_csv_value(row, "case_id", row_number),
                    label=_required_csv_value(row, "label", row_number),
                    factor=_required_csv_value(row, "factor", row_number),
                    correct=_parse_bool(
                        _required_csv_value(row, "correct", row_number),
                        row_number,
                    ),
                )
            )
        else:
            raise ValueError(f"Row {row_number}: unknown metric {metric!r}")
    return HumanRatingsFile(
        depth_ratings=depth_ratings,
        category_ratings=category_ratings,
    )


def validate_complete_human_ratings(
    ratings: HumanRatingsFile,
    packet: HumanCalibrationPacket,
    *,
    rater_ids: Sequence[str] | None = None,
) -> HumanRatingsValidationReport:
    expected_depth = {
        _depth_key(assigned_rater_id(packet, item.case_id), item.case_id, output.label)
        for item in packet.items
        for output in item.outputs
        if rater_ids is None or assigned_rater_id(packet, item.case_id) in rater_ids
    }
    expected_category = {
        _category_key(
            assigned_rater_id(packet, item.case_id),
            item.case_id,
            output.label,
            target["factor"],
        )
        for item in packet.items
        for output in item.outputs
        for target in item.category_targets
        if rater_ids is None or assigned_rater_id(packet, item.case_id) in rater_ids
    }
    observed_depth_counts = _counts(
        _depth_key(r.rater_id, r.case_id, r.label) for r in ratings.depth_ratings
    )
    observed_category_counts = _counts(
        _category_key(r.rater_id, r.case_id, r.label, r.factor)
        for r in ratings.category_ratings
    )
    observed_depth = set(observed_depth_counts)
    observed_category = set(observed_category_counts)

    missing_depth = sorted(expected_depth - observed_depth)
    missing_category = sorted(expected_category - observed_category)
    duplicate_depth = sorted(
        key for key, count in observed_depth_counts.items() if count > 1
    )
    duplicate_category = sorted(
        key for key, count in observed_category_counts.items() if count > 1
    )
    unexpected_depth = sorted(observed_depth - expected_depth)
    unexpected_category = sorted(observed_category - expected_category)

    return HumanRatingsValidationReport(
        is_complete=not (
            missing_depth
            or missing_category
            or duplicate_depth
            or duplicate_category
            or unexpected_depth
            or unexpected_category
        ),
        expected_depth_count=len(expected_depth),
        observed_depth_count=len(ratings.depth_ratings),
        expected_category_count=len(expected_category),
        observed_category_count=len(ratings.category_ratings),
        missing_depth_ratings=missing_depth,
        missing_category_ratings=missing_category,
        duplicate_depth_ratings=duplicate_depth,
        duplicate_category_ratings=duplicate_category,
        unexpected_depth_ratings=unexpected_depth,
        unexpected_category_ratings=unexpected_category,
    )


def validate_human_ratings_template(
    template: HumanRatingsTemplate,
    packet: HumanCalibrationPacket,
    *,
    rater_ids: Sequence[str] | None = None,
) -> HumanRatingsTemplateValidationReport:
    """Validate that a blank rating template exactly matches the packet."""
    expected_depth = {
        _depth_key(assigned_rater_id(packet, item.case_id), item.case_id, output.label)
        for item in packet.items
        for output in item.outputs
        if rater_ids is None or assigned_rater_id(packet, item.case_id) in rater_ids
    }
    expected_category_context = {
        _category_key(
            assigned_rater_id(packet, item.case_id),
            item.case_id,
            output.label,
            target["factor"],
        ): (
            target.get("correct_category", ""),
            target.get("rationale", ""),
        )
        for item in packet.items
        for output in item.outputs
        for target in item.category_targets
        if rater_ids is None or assigned_rater_id(packet, item.case_id) in rater_ids
    }
    expected_category = set(expected_category_context)

    observed_depth_counts = _counts(
        _depth_key(r.rater_id, r.case_id, r.label) for r in template.depth_ratings
    )
    observed_category_counts = _counts(
        _category_key(r.rater_id, r.case_id, r.label, r.factor)
        for r in template.category_ratings
    )
    observed_depth = set(observed_depth_counts)
    observed_category = set(observed_category_counts)

    context_mismatches: list[str] = []
    for rating in template.category_ratings:
        key = _category_key(rating.rater_id, rating.case_id, rating.label, rating.factor)
        expected = expected_category_context.get(key)
        if expected is None:
            continue
        expected_category_value, expected_rationale = expected
        observed_category_value = rating.correct_category or ""
        observed_rationale = rating.rationale or ""
        if (
            observed_category_value != expected_category_value
            or observed_rationale != expected_rationale
        ):
            context_mismatches.append(
                f"{key}: expected category={expected_category_value!r}, "
                f"rationale={expected_rationale!r}; observed "
                f"category={observed_category_value!r}, rationale={observed_rationale!r}"
            )

    prefilled_depth = sorted(
        _depth_key(r.rater_id, r.case_id, r.label)
        for r in template.depth_ratings
        if r.analytical_depth_score is not None
    )
    prefilled_category = sorted(
        _category_key(r.rater_id, r.case_id, r.label, r.factor)
        for r in template.category_ratings
        if r.correct is not None
    )

    missing_depth = sorted(expected_depth - observed_depth)
    missing_category = sorted(expected_category - observed_category)
    duplicate_depth = sorted(
        key for key, count in observed_depth_counts.items() if count > 1
    )
    duplicate_category = sorted(
        key for key, count in observed_category_counts.items() if count > 1
    )
    unexpected_depth = sorted(observed_depth - expected_depth)
    unexpected_category = sorted(observed_category - expected_category)
    is_valid = not (
        missing_depth
        or missing_category
        or duplicate_depth
        or duplicate_category
        or unexpected_depth
        or unexpected_category
        or context_mismatches
        or prefilled_depth
        or prefilled_category
    )

    return HumanRatingsTemplateValidationReport(
        is_valid=is_valid,
        expected_depth_count=len(expected_depth),
        observed_depth_count=len(template.depth_ratings),
        expected_category_count=len(expected_category),
        observed_category_count=len(template.category_ratings),
        missing_depth_rows=missing_depth,
        missing_category_rows=missing_category,
        duplicate_depth_rows=duplicate_depth,
        duplicate_category_rows=duplicate_category,
        unexpected_depth_rows=unexpected_depth,
        unexpected_category_rows=unexpected_category,
        category_context_mismatches=sorted(context_mismatches),
        prefilled_depth_rows=prefilled_depth,
        prefilled_category_rows=prefilled_category,
    )


def _answer_key_lookup(
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> dict[tuple[str, str], str]:
    return {(entry.case_id, entry.label): entry.system_id for entry in answer_key}


def _judged_lookup(
    judged_records: Sequence[JudgedModelResponse],
) -> dict[tuple[str, str], JudgedModelResponse]:
    return {
        (record.response.case_id, record.response.system_id): record
        for record in judged_records
        if record.judge is not None
    }


def _llm_depth_lookup(
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> dict[tuple[str, str], int]:
    by_system = _judged_lookup(judged_records)
    depths: dict[tuple[str, str], int] = {}
    for entry in answer_key:
        judged = by_system.get((entry.case_id, entry.system_id))
        if judged is None or judged.judge is None:
            continue
        depths[(entry.case_id, entry.label)] = discretize_depth_score(
            judged.judge.analytical_depth_score
        )
    return depths


def _llm_category_lookup(
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> dict[tuple[str, str, str], bool]:
    by_system = _judged_lookup(judged_records)
    judgments: dict[tuple[str, str, str], bool] = {}
    for entry in answer_key:
        judged = by_system.get((entry.case_id, entry.system_id))
        if judged is None or judged.judge is None:
            continue
        for category_judgment in judged.judge.category_judgments:
            judgments[(entry.case_id, entry.label, category_judgment.factor)] = (
                category_judgment.correct
            )
    return judgments


def fleiss_for_human_depth(
    ratings: Sequence[HumanDepthRating],
) -> KappaSummary | None:
    matrix = _complete_rating_matrix(
        ((rating.case_id, rating.label), rating.rater_id, rating.analytical_depth_score)
        for rating in ratings
    )
    if not matrix:
        return None
    return KappaSummary(
        metric="analytical_depth",
        comparison="human_fleiss",
        n_items=len(matrix),
        kappa=fleiss_kappa(matrix, labels=[1, 2, 3]),
    )


def cohen_depth_by_rater(
    ratings: Sequence[HumanDepthRating],
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> list[KappaSummary]:
    llm_scores = _llm_depth_lookup(judged_records, answer_key)
    by_rater: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rating in ratings:
        llm_score = llm_scores.get((rating.case_id, rating.label))
        if llm_score is None:
            continue
        by_rater[rating.rater_id].append((rating.analytical_depth_score, llm_score))
    return [
        KappaSummary(
            metric="analytical_depth",
            comparison=f"{rater_id}_vs_llm",
            n_items=len(pairs),
            kappa=cohen_kappa(
                [human for human, _ in pairs],
                [llm for _, llm in pairs],
                labels=[1, 2, 3],
            )
            if pairs
            else None,
        )
        for rater_id, pairs in sorted(by_rater.items())
    ]


def weighted_cohen_depth_by_rater(
    ratings: Sequence[HumanDepthRating],
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> list[KappaSummary]:
    """Compute quadratic-weighted human-vs-LLM Cohen kappa per rater."""
    llm_scores = _llm_depth_lookup(judged_records, answer_key)
    by_rater: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rating in ratings:
        llm_score = llm_scores.get((rating.case_id, rating.label))
        if llm_score is None:
            continue
        by_rater[rating.rater_id].append((rating.analytical_depth_score, llm_score))
    return [
        KappaSummary(
            metric="analytical_depth",
            comparison=f"{rater_id}_vs_llm_quadratic_weighted",
            n_items=len(pairs),
            kappa=weighted_cohen_kappa(
                [human for human, _ in pairs],
                [llm for _, llm in pairs],
                labels=[1, 2, 3],
                weighting="quadratic",
            )
            if pairs
            else None,
        )
        for rater_id, pairs in sorted(by_rater.items())
    ]


def pooled_cohen_depth(
    ratings: Sequence[HumanDepthRating],
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> KappaSummary | None:
    """Compute one pooled human-vs-LLM Cohen kappa over all depth ratings."""
    llm_scores = _llm_depth_lookup(judged_records, answer_key)
    pairs = [
        (rating.analytical_depth_score, llm_score)
        for rating in ratings
        if (llm_score := llm_scores.get((rating.case_id, rating.label))) is not None
    ]
    if not pairs:
        return None
    return KappaSummary(
        metric="analytical_depth",
        comparison="pooled_humans_vs_llm",
        n_items=len(pairs),
        kappa=cohen_kappa(
            [human for human, _ in pairs],
            [llm for _, llm in pairs],
            labels=[1, 2, 3],
        ),
    )


def pooled_weighted_cohen_depth(
    ratings: Sequence[HumanDepthRating],
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> KappaSummary | None:
    """Compute pooled quadratic-weighted Cohen kappa for ordered depth ratings."""
    llm_scores = _llm_depth_lookup(judged_records, answer_key)
    pairs = [
        (rating.analytical_depth_score, llm_score)
        for rating in ratings
        if (llm_score := llm_scores.get((rating.case_id, rating.label))) is not None
    ]
    if not pairs:
        return None
    return KappaSummary(
        metric="analytical_depth",
        comparison="pooled_humans_vs_llm_quadratic_weighted",
        n_items=len(pairs),
        kappa=weighted_cohen_kappa(
            [human for human, _ in pairs],
            [llm for _, llm in pairs],
            labels=[1, 2, 3],
            weighting="quadratic",
        ),
    )


def fleiss_for_depth_with_llm(
    ratings: Sequence[HumanDepthRating],
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> KappaSummary | None:
    """Compute Fleiss kappa with the LLM judge treated as an extra rater."""
    llm_scores = _llm_depth_lookup(judged_records, answer_key)
    rows = [
        ((rating.case_id, rating.label), rating.rater_id, rating.analytical_depth_score)
        for rating in ratings
    ]
    rows.extend(
        ((case_id, label), "llm_judge", score)
        for (case_id, label), score in llm_scores.items()
    )
    matrix = _complete_rating_matrix(rows)
    if not matrix:
        return None
    return KappaSummary(
        metric="analytical_depth",
        comparison="human_llm_fleiss",
        n_items=len(matrix),
        kappa=fleiss_kappa(matrix, labels=[1, 2, 3]),
    )


def fleiss_for_human_category(
    ratings: Sequence[HumanCategoryRating],
) -> KappaSummary | None:
    matrix = _complete_rating_matrix(
        (
            (rating.case_id, rating.label, rating.factor),
            rating.rater_id,
            rating.correct,
        )
        for rating in ratings
    )
    if not matrix:
        return None
    return KappaSummary(
        metric="categorization_accuracy",
        comparison="human_fleiss",
        n_items=len(matrix),
        kappa=fleiss_kappa(matrix, labels=[False, True]),
    )


def cohen_category_by_rater(
    ratings: Sequence[HumanCategoryRating],
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> list[KappaSummary]:
    llm_scores = _llm_category_lookup(judged_records, answer_key)
    by_rater: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for rating in ratings:
        llm_correct = llm_scores.get((rating.case_id, rating.label, rating.factor))
        if llm_correct is None:
            continue
        by_rater[rating.rater_id].append((rating.correct, llm_correct))
    return [
        KappaSummary(
            metric="categorization_accuracy",
            comparison=f"{rater_id}_vs_llm",
            n_items=len(pairs),
            kappa=cohen_kappa(
                [human for human, _ in pairs],
                [llm for _, llm in pairs],
                labels=[False, True],
            )
            if pairs
            else None,
        )
        for rater_id, pairs in sorted(by_rater.items())
    ]


def pooled_cohen_category(
    ratings: Sequence[HumanCategoryRating],
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> KappaSummary | None:
    """Compute one pooled human-vs-LLM Cohen kappa over all category ratings."""
    llm_scores = _llm_category_lookup(judged_records, answer_key)
    pairs: list[tuple[bool, bool]] = []
    for rating in ratings:
        llm_correct = llm_scores.get((rating.case_id, rating.label, rating.factor))
        if llm_correct is None:
            continue
        pairs.append((rating.correct, llm_correct))
    if not pairs:
        return None
    return KappaSummary(
        metric="categorization_accuracy",
        comparison="pooled_humans_vs_llm",
        n_items=len(pairs),
        kappa=cohen_kappa(
            [human for human, _ in pairs],
            [llm for _, llm in pairs],
            labels=[False, True],
        ),
    )


def fleiss_for_category_with_llm(
    ratings: Sequence[HumanCategoryRating],
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> KappaSummary | None:
    """Compute category Fleiss kappa with the LLM judge as an extra rater."""
    llm_scores = _llm_category_lookup(judged_records, answer_key)
    rows = [
        ((rating.case_id, rating.label, rating.factor), rating.rater_id, rating.correct)
        for rating in ratings
    ]
    rows.extend(
        ((case_id, label, factor), "llm_judge", correct)
        for (case_id, label, factor), correct in llm_scores.items()
    )
    matrix = _complete_rating_matrix(rows)
    if not matrix:
        return None
    return KappaSummary(
        metric="categorization_accuracy",
        comparison="human_llm_fleiss",
        n_items=len(matrix),
        kappa=fleiss_kappa(matrix, labels=[False, True]),
    )


def compute_human_irr_report(
    ratings: HumanRatingsFile,
    judged_records: Sequence[JudgedModelResponse],
    answer_key: Sequence[HumanAnswerKeyEntry],
) -> HumanIrrReport:
    """Compare human ratings against the LLM judge.

    When each case is rated by only one human rater, human-only Fleiss kappa is
    omitted because there is no multi-rater overlap on the same items.
    """
    depth_has_overlap = _has_multi_rater_overlap(
        ((rating.case_id, rating.label), rating.rater_id) for rating in ratings.depth_ratings
    )
    category_has_overlap = _has_multi_rater_overlap(
        (
            (rating.case_id, rating.label, rating.factor),
            rating.rater_id,
        )
        for rating in ratings.category_ratings
    )
    depth_fleiss_with_llm = (
        fleiss_for_depth_with_llm(ratings.depth_ratings, judged_records, answer_key)
        if depth_has_overlap
        else None
    )
    category_fleiss_with_llm = (
        fleiss_for_category_with_llm(
            ratings.category_ratings, judged_records, answer_key
        )
        if category_has_overlap
        else None
    )
    return HumanIrrReport(
        depth_fleiss=(
            fleiss_for_human_depth(ratings.depth_ratings)
            if depth_has_overlap
            else None
        ),
        depth_fleiss_with_llm=depth_fleiss_with_llm,
        depth_cohen_pooled=pooled_cohen_depth(
            ratings.depth_ratings, judged_records, answer_key
        ),
        depth_cohen_by_rater=cohen_depth_by_rater(
            ratings.depth_ratings, judged_records, answer_key
        ),
        depth_weighted_cohen_pooled=pooled_weighted_cohen_depth(
            ratings.depth_ratings, judged_records, answer_key
        ),
        depth_weighted_cohen_by_rater=weighted_cohen_depth_by_rater(
            ratings.depth_ratings, judged_records, answer_key
        ),
        category_fleiss=(
            fleiss_for_human_category(ratings.category_ratings)
            if category_has_overlap
            else None
        ),
        category_fleiss_with_llm=category_fleiss_with_llm,
        category_cohen_pooled=pooled_cohen_category(
            ratings.category_ratings, judged_records, answer_key
        ),
        category_cohen_by_rater=cohen_category_by_rater(
            ratings.category_ratings, judged_records, answer_key
        ),
    )


def _has_multi_rater_overlap(
    rows: Iterable[tuple[tuple[str, ...], str]],
) -> bool:
    item_raters: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for item_key, rater_id in rows:
        item_raters[item_key].add(rater_id)
    return any(len(raters) > 1 for raters in item_raters.values())


def _complete_rating_matrix(
    rows: Iterable[tuple[tuple[str, ...], str, object]],
) -> list[list[object]]:
    grouped: dict[tuple[str, ...], dict[str, object]] = defaultdict(dict)
    raters: set[str] = set()
    for item_key, rater_id, value in rows:
        grouped[item_key][rater_id] = value
        raters.add(rater_id)
    ordered_raters = sorted(raters)
    return [
        [ratings[rater_id] for rater_id in ordered_raters]
        for ratings in grouped.values()
        if all(rater_id in ratings for rater_id in ordered_raters)
    ]


def _depth_key(rater_id: str, case_id: str, label: str) -> str:
    return f"{rater_id}|{case_id}|{label}"


def _category_key(rater_id: str, case_id: str, label: str, factor: str) -> str:
    return f"{rater_id}|{case_id}|{label}|{factor}"


def _counts(keys: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for key in keys:
        counts[key] += 1
    return counts


def _required_csv_value(
    row: Mapping[str, str],
    field: str,
    row_number: int,
) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"Row {row_number}: missing required field {field!r}")
    return value


def _parse_depth_score(row: Mapping[str, str], row_number: int) -> int:
    raw_score = _required_csv_value(row, "analytical_depth_score", row_number)
    try:
        return int(raw_score)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: analytical_depth_score must be 1, 2, or 3"
        ) from exc


def _parse_bool(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(
        f"Row {row_number}: correct must be true/false, yes/no, or 1/0"
    )
