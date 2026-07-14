"""Contracts and helpers for gold-standard control-group experiments."""

import hashlib
import math
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from mascan.eval.gold_judge import GoldJudgeResult

CONTROL_GROUP_SYSTEMS = (
    "mascan",
    "zero_shot_same_model",
    "frontier_model",
)


class TokenUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated: bool = False
    cost_usd: float | None = Field(default=None, ge=0.0)


class ModelResponseRecord(BaseModel):
    case_id: str
    system_id: str
    model: str
    prompt_sha256: str | None = None
    generation_config: dict[str, Any] = Field(default_factory=dict)
    response_text: str = ""
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float | None = Field(default=None, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None


class JudgedModelResponse(BaseModel):
    response: ModelResponseRecord
    judge: GoldJudgeResult | None = None
    error: str | None = None


class SystemMetricSummary(BaseModel):
    system_id: str
    n: int
    total_quality_points: float | None = None
    mean_analytical_depth: float | None = None
    mean_categorization_accuracy: float | None = None
    mean_combined_quality: float | None = None
    total_missing_gold_claims: int | None = None
    mean_missing_gold_claims: float | None = None
    total_unsupported_or_wrong_claims: int | None = None
    mean_unsupported_or_wrong_claims: float | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    quality_per_1k_tokens: float | None = None
    quality_per_usd: float | None = None


class MetricPairRecord(BaseModel):
    case_id: str
    treatment_score: float
    control_score: float
    difference: float


def estimate_token_count(text: str) -> int:
    """Small deterministic fallback when provider token usage is unavailable."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def estimate_token_usage(prompt: str, response_text: str) -> TokenUsage:
    prompt_tokens = estimate_token_count(prompt)
    completion_tokens = estimate_token_count(response_text)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated=True,
    )


def prompt_sha256(prompt: str) -> str:
    """Stable hash proving which frozen prompt generated a response."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def combined_quality_score(judge: GoldJudgeResult) -> float:
    """Normalize depth and accuracy to [0, 1] and average them."""
    normalized_depth = judge.analytical_depth_score / 3.0
    return round((normalized_depth + judge.categorization_accuracy) / 2.0, 6)


def response_lookup(
    records: Iterable[ModelResponseRecord],
) -> dict[tuple[str, str], ModelResponseRecord]:
    lookup: dict[tuple[str, str], ModelResponseRecord] = {}
    for record in records:
        lookup[(record.case_id, record.system_id)] = record
    return lookup


def metric_pairs(
    judged_records: Sequence[JudgedModelResponse],
    *,
    treatment_system: str,
    control_system: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    """Return paired scores aligned by case_id for significance testing."""
    pairs = metric_pair_records(
        judged_records,
        treatment_system=treatment_system,
        control_system=control_system,
        metric=metric,
    )
    return (
        [pair.treatment_score for pair in pairs],
        [pair.control_score for pair in pairs],
    )


def metric_pair_records(
    judged_records: Sequence[JudgedModelResponse],
    *,
    treatment_system: str,
    control_system: str,
    metric: str,
) -> list[MetricPairRecord]:
    """Return paired scores with case IDs for statistical audit trails."""
    by_key = {
        (record.response.case_id, record.response.system_id): record
        for record in judged_records
        if record.judge is not None
    }
    case_ids = sorted(
        {
            case_id
            for case_id, system_id in by_key
            if system_id in {treatment_system, control_system}
        }
    )
    pairs: list[MetricPairRecord] = []
    for case_id in case_ids:
        treatment_record = by_key.get((case_id, treatment_system))
        control_record = by_key.get((case_id, control_system))
        if treatment_record is None or control_record is None:
            continue
        if treatment_record.judge is None or control_record.judge is None:
            continue
        treatment_score = _metric_value(treatment_record.judge, metric)
        control_score = _metric_value(control_record.judge, metric)
        pairs.append(
            MetricPairRecord(
                case_id=case_id,
                treatment_score=treatment_score,
                control_score=control_score,
                difference=round(treatment_score - control_score, 6),
            )
        )
    return pairs


def _metric_value(judge: GoldJudgeResult, metric: str) -> float:
    if metric == "analytical_depth":
        return judge.analytical_depth_score
    if metric == "categorization_accuracy":
        return judge.categorization_accuracy
    if metric == "combined_quality":
        return combined_quality_score(judge)
    raise ValueError(f"Unknown metric: {metric}")


def summarize_system(
    system_id: str,
    judged_records: Sequence[JudgedModelResponse],
) -> SystemMetricSummary:
    relevant = [
        record
        for record in judged_records
        if record.response.system_id == system_id and record.judge is not None
    ]
    if not relevant:
        return SystemMetricSummary(system_id=system_id, n=0)

    depth_scores = [record.judge.analytical_depth_score for record in relevant]
    accuracy_scores = [record.judge.categorization_accuracy for record in relevant]
    combined_scores = [combined_quality_score(record.judge) for record in relevant]
    missing_claim_counts = [
        len(record.judge.missing_gold_claims) for record in relevant
    ]
    unsupported_claim_counts = [
        len(record.judge.unsupported_or_wrong_claims) for record in relevant
    ]
    token_totals = [
        record.response.token_usage.total_tokens
        for record in relevant
        if record.response.token_usage.total_tokens is not None
    ]
    costs = [
        record.response.token_usage.cost_usd
        for record in relevant
        if record.response.token_usage.cost_usd is not None
    ]
    total_tokens = sum(token_totals) if token_totals else None
    total_cost = round(sum(costs), 6) if costs else None
    mean_combined = sum(combined_scores) / len(combined_scores)
    total_quality = round(sum(combined_scores), 6)

    return SystemMetricSummary(
        system_id=system_id,
        n=len(relevant),
        total_quality_points=total_quality,
        mean_analytical_depth=round(sum(depth_scores) / len(depth_scores), 6),
        mean_categorization_accuracy=round(
            sum(accuracy_scores) / len(accuracy_scores), 6
        ),
        mean_combined_quality=round(mean_combined, 6),
        total_missing_gold_claims=sum(missing_claim_counts),
        mean_missing_gold_claims=round(
            sum(missing_claim_counts) / len(missing_claim_counts), 6
        ),
        total_unsupported_or_wrong_claims=sum(unsupported_claim_counts),
        mean_unsupported_or_wrong_claims=round(
            sum(unsupported_claim_counts) / len(unsupported_claim_counts), 6
        ),
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        quality_per_1k_tokens=(
            round(total_quality / total_tokens * 1000, 8)
            if total_tokens
            else None
        ),
        quality_per_usd=(
            round(total_quality / total_cost, 8)
            if total_cost
            else None
        ),
    )
