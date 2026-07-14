"""Analysis helpers for judged gold-standard experiment outputs."""

from collections.abc import Sequence
from statistics import median

from pydantic import BaseModel, Field

from mascan.eval.gold_experiment import (
    JudgedModelResponse,
    MetricPairRecord,
    SystemMetricSummary,
    combined_quality_score,
    metric_pair_records,
    summarize_system,
)
from mascan.eval.stats import Alternative, PairedTestResult, compare_paired_scores


class SystemComparison(BaseModel):
    treatment_system: str
    control_system: str
    metric: str
    paired_cases: int
    paired_case_ids: list[str] = Field(default_factory=list)
    treatment_scores: list[float]
    control_scores: list[float]
    paired_differences: list[float] = Field(default_factory=list)
    mean_difference: float | None = None
    median_difference: float | None = None
    score_pairs: list[MetricPairRecord] = Field(default_factory=list)
    result: PairedTestResult


class CaseTraceRecord(BaseModel):
    case_id: str
    system_id: str
    model: str
    prompt_sha256: str | None = None
    analytical_depth_score: float | None = None
    categorization_accuracy: float | None = None
    combined_quality: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_tokens: bool
    cost_usd: float | None = None
    latency_seconds: float | None = None
    quality_per_1k_tokens: float | None = None
    quality_per_usd: float | None = None
    response_claim_count: int | None = None
    missing_gold_claim_count: int | None = None
    unsupported_or_wrong_claim_count: int | None = None
    category_targets_evaluated: int | None = None
    category_targets_correct: int | None = None
    error: str | None = None


def summarize_systems(
    judged_records: Sequence[JudgedModelResponse],
) -> list[SystemMetricSummary]:
    system_ids = sorted({record.response.system_id for record in judged_records})
    return [summarize_system(system_id, judged_records) for system_id in system_ids]


def case_trace_records(
    judged_records: Sequence[JudgedModelResponse],
) -> list[CaseTraceRecord]:
    """Flatten judged records into per-case token/cost/quality trace rows."""
    rows: list[CaseTraceRecord] = []
    for record in sorted(
        judged_records,
        key=lambda item: (item.response.case_id, item.response.system_id),
    ):
        response = record.response
        usage = response.token_usage
        judge = record.judge
        combined = combined_quality_score(judge) if judge is not None else None
        rows.append(
            CaseTraceRecord(
                case_id=response.case_id,
                system_id=response.system_id,
                model=response.model,
                prompt_sha256=response.prompt_sha256,
                analytical_depth_score=(
                    judge.analytical_depth_score if judge is not None else None
                ),
                categorization_accuracy=(
                    judge.categorization_accuracy if judge is not None else None
                ),
                combined_quality=combined,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_tokens=usage.estimated,
                cost_usd=usage.cost_usd,
                latency_seconds=response.latency_seconds,
                quality_per_1k_tokens=_quality_per_1k_tokens(
                    combined,
                    usage.total_tokens,
                ),
                quality_per_usd=_quality_per_usd(combined, usage.cost_usd),
                response_claim_count=(
                    judge.response_claim_count if judge is not None else None
                ),
                missing_gold_claim_count=(
                    len(judge.missing_gold_claims) if judge is not None else None
                ),
                unsupported_or_wrong_claim_count=(
                    len(judge.unsupported_or_wrong_claims)
                    if judge is not None
                    else None
                ),
                category_targets_evaluated=(
                    judge.category_targets_evaluated if judge is not None else None
                ),
                category_targets_correct=(
                    judge.category_targets_correct if judge is not None else None
                ),
                error=record.error or response.error,
            )
        )
    return rows


def compare_systems(
    judged_records: Sequence[JudgedModelResponse],
    *,
    treatment_system: str,
    control_system: str,
    metric: str,
    assume_normal: bool | None = None,
    normality_alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> SystemComparison:
    pairs = metric_pair_records(
        judged_records,
        treatment_system=treatment_system,
        control_system=control_system,
        metric=metric,
    )
    treatment_scores = [pair.treatment_score for pair in pairs]
    control_scores = [pair.control_score for pair in pairs]
    if not treatment_scores:
        raise ValueError(
            f"No paired scores found for {treatment_system} vs {control_system}"
        )
    result = compare_paired_scores(
        treatment_scores,
        control_scores,
        assume_normal=assume_normal,
        alpha=normality_alpha,
        alternative=alternative,
    )
    differences = [pair.difference for pair in pairs]
    return SystemComparison(
        treatment_system=treatment_system,
        control_system=control_system,
        metric=metric,
        paired_cases=len(treatment_scores),
        paired_case_ids=[pair.case_id for pair in pairs],
        treatment_scores=treatment_scores,
        control_scores=control_scores,
        paired_differences=differences,
        mean_difference=round(sum(differences) / len(differences), 6),
        median_difference=round(float(median(differences)), 6),
        score_pairs=pairs,
        result=result,
    )


def _quality_per_1k_tokens(
    combined_quality: float | None,
    total_tokens: int | None,
) -> float | None:
    if combined_quality is None or not total_tokens:
        return None
    return round(combined_quality / total_tokens * 1000, 8)


def _quality_per_usd(
    combined_quality: float | None,
    cost_usd: float | None,
) -> float | None:
    if combined_quality is None or not cost_usd:
        return None
    return round(combined_quality / cost_usd, 8)
