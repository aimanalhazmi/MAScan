"""Paired statistical tests for comparing evaluation systems."""

import math
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

Alternative = Literal["two-sided", "greater", "less"]
TestName = Literal["paired_t_test", "wilcoxon_signed_rank"]


class PairedTestResult(BaseModel):
    test_name: TestName
    alternative: Alternative
    n: int = Field(..., ge=0)
    statistic: float
    p_value: float = Field(..., ge=0.0, le=1.0)
    mean_difference: float
    effect_size_name: str | None = None
    effect_size: float | None = None
    normality_method: str | None = None
    normality_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    normality_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    test_selection_reason: str | None = None


def _validate_pairs(
    treatment: Sequence[float], control: Sequence[float]
) -> list[float]:
    if len(treatment) != len(control):
        raise ValueError("treatment and control must have the same length")
    if not treatment:
        raise ValueError("at least one paired observation is required")
    return [float(a) - float(b) for a, b in zip(treatment, control, strict=True)]


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        a * math.log(x)
        + b * math.log1p(-x)
        - math.lgamma(a)
        - math.lgamma(b)
        + math.lgamma(a + b)
    )
    front = math.exp(log_front)

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(t_value: float, df: int) -> float:
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(t_value):
        return 1.0 if t_value > 0 else 0.0

    x = df / (df + t_value * t_value)
    ibeta = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    if t_value >= 0:
        return 1.0 - 0.5 * ibeta
    return 0.5 * ibeta


def _p_from_symmetric_cdf(
    statistic: float, cdf: float, alternative: Alternative
) -> float:
    if alternative == "greater":
        return 1.0 - cdf
    if alternative == "less":
        return cdf
    return min(1.0, 2.0 * min(cdf, 1.0 - cdf))


def paired_t_test(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
) -> PairedTestResult:
    """Run a paired t-test on treatment-control differences."""
    differences = _validate_pairs(treatment, control)
    n = len(differences)
    mean_diff = sum(differences) / n
    effect_size: float | None = None
    if n == 1:
        statistic = math.inf if mean_diff > 0 else -math.inf if mean_diff < 0 else 0.0
        p_value = 0.0 if mean_diff != 0 else 1.0
    else:
        variance = sum((x - mean_diff) ** 2 for x in differences) / (n - 1)
        if variance > 0.0:
            effect_size = mean_diff / math.sqrt(variance)
        if variance == 0.0:
            statistic = (
                math.inf if mean_diff > 0 else -math.inf if mean_diff < 0 else 0.0
            )
            p_value = 0.0 if mean_diff != 0 else 1.0
        else:
            statistic = mean_diff / math.sqrt(variance / n)
            cdf = _student_t_cdf(statistic, n - 1)
            p_value = _p_from_symmetric_cdf(statistic, cdf, alternative)

    return PairedTestResult(
        test_name="paired_t_test",
        alternative=alternative,
        n=n,
        statistic=round(statistic, 6) if math.isfinite(statistic) else statistic,
        p_value=round(p_value, 6),
        mean_difference=round(mean_diff, 6),
        effect_size_name="cohen_dz",
        effect_size=round(effect_size, 6) if effect_size is not None else None,
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average_rank
        i = j
    return ranks


def _exact_signed_rank_p_value(
    observed_positive_rank_sum: float,
    ranks: Sequence[float],
    alternative: Alternative,
) -> float:
    scaled_ranks = [int(round(rank * 2)) for rank in ranks]
    observed = int(round(observed_positive_rank_sum * 2))
    total = sum(scaled_ranks)

    counts: Counter[int] = Counter({0: 1})
    for rank in scaled_ranks:
        next_counts = counts.copy()
        for score, count in counts.items():
            next_counts[score + rank] += count
        counts = next_counts

    outcomes = 2 ** len(scaled_ranks)
    if alternative == "greater":
        favourable = sum(count for score, count in counts.items() if score >= observed)
    elif alternative == "less":
        favourable = sum(count for score, count in counts.items() if score <= observed)
    else:
        observed_extremity = min(observed, total - observed)
        favourable = sum(
            count
            for score, count in counts.items()
            if min(score, total - score) <= observed_extremity
        )
    return float(favourable / outcomes)


def wilcoxon_signed_rank_test(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
) -> PairedTestResult:
    """Run an exact Wilcoxon signed-rank test for paired score differences."""
    raw_differences = _validate_pairs(treatment, control)
    differences = [diff for diff in raw_differences if diff != 0.0]
    if not differences:
        return PairedTestResult(
            test_name="wilcoxon_signed_rank",
            alternative=alternative,
            n=0,
            statistic=0.0,
            p_value=1.0,
            mean_difference=0.0,
            effect_size_name="rank_biserial_correlation",
            effect_size=0.0,
        )

    abs_differences = [abs(diff) for diff in differences]
    ranks = _average_ranks(abs_differences)
    positive_rank_sum = sum(
        rank for diff, rank in zip(differences, ranks, strict=True) if diff > 0
    )
    total_rank_sum = sum(ranks)
    negative_rank_sum = total_rank_sum - positive_rank_sum
    effect_size = (
        (positive_rank_sum - negative_rank_sum) / total_rank_sum
        if total_rank_sum
        else 0.0
    )
    statistic = (
        positive_rank_sum
        if alternative != "two-sided"
        else min(positive_rank_sum, total_rank_sum - positive_rank_sum)
    )
    p_value = _exact_signed_rank_p_value(positive_rank_sum, ranks, alternative)

    return PairedTestResult(
        test_name="wilcoxon_signed_rank",
        alternative=alternative,
        n=len(differences),
        statistic=round(statistic, 6),
        p_value=round(p_value, 6),
        mean_difference=round(sum(raw_differences) / len(raw_differences), 6),
        effect_size_name="rank_biserial_correlation",
        effect_size=round(effect_size, 6),
    )


def _try_shapiro_p_value(values: Sequence[float]) -> float | None:
    try:
        from scipy import stats as scipy_stats
    except ModuleNotFoundError:
        return None
    result = scipy_stats.shapiro(values)
    return float(result.pvalue)


def compare_paired_scores(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    alternative: Alternative = "two-sided",
    assume_normal: bool | None = None,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Choose a paired significance test for treatment-control scores.

    If assume_normal is None, the function tries SciPy's Shapiro-Wilk normality
    test on paired differences. When SciPy is unavailable, it defaults to the
    non-parametric Wilcoxon test and records that normality was not tested.
    """
    differences = _validate_pairs(treatment, control)
    normality_method: str | None = None
    normality_p_value: float | None = None

    if assume_normal is None:
        normality_p_value = _try_shapiro_p_value(differences)
        if normality_p_value is None:
            normality_method = "not_available_scipy_missing"
            selection_reason = (
                "SciPy Shapiro-Wilk normality test unavailable; selected "
                "Wilcoxon signed-rank as the conservative non-parametric default."
            )
            selected = wilcoxon_signed_rank_test(
                treatment, control, alternative=alternative
            )
        else:
            normality_method = "shapiro_wilk"
            if normality_p_value >= alpha:
                selection_reason = (
                    f"Shapiro-Wilk p-value {normality_p_value:.6f} >= "
                    f"alpha {alpha}; selected paired t-test."
                )
                selected = paired_t_test(treatment, control, alternative=alternative)
            else:
                selection_reason = (
                    f"Shapiro-Wilk p-value {normality_p_value:.6f} < "
                    f"alpha {alpha}; selected Wilcoxon signed-rank."
                )
                selected = wilcoxon_signed_rank_test(
                    treatment, control, alternative=alternative
                )
    elif assume_normal:
        normality_method = "assumed_normal"
        selection_reason = "Normality assumed by caller; selected paired t-test."
        selected = paired_t_test(treatment, control, alternative=alternative)
    else:
        normality_method = "assumed_non_parametric"
        selection_reason = (
            "Normality not assumed by caller; selected Wilcoxon signed-rank."
        )
        selected = wilcoxon_signed_rank_test(
            treatment, control, alternative=alternative
        )

    return selected.model_copy(
        update={
            "normality_method": normality_method,
            "normality_alpha": alpha,
            "normality_p_value": (
                round(normality_p_value, 6)
                if normality_p_value is not None
                else None
            ),
            "test_selection_reason": selection_reason,
        }
    )
