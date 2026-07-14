import math

import pytest

from mascan.eval.stats import (
    compare_paired_scores,
    paired_t_test,
    wilcoxon_signed_rank_test,
)


def test_paired_t_test_returns_expected_statistic():
    result = paired_t_test([2, 4, 6, 8, 10], [1, 2, 3, 4, 5])

    assert result.test_name == "paired_t_test"
    assert result.n == 5
    assert result.mean_difference == 3.0
    assert result.statistic == pytest.approx(4.242641, abs=0.000001)
    assert 0.0 <= result.p_value <= 1.0
    assert result.effect_size_name == "cohen_dz"
    assert result.effect_size == pytest.approx(1.897367, abs=0.000001)


def test_paired_t_test_handles_zero_variance_difference():
    result = paired_t_test([2, 3, 4], [1, 2, 3])

    assert math.isinf(result.statistic)
    assert result.p_value == 0.0
    assert result.effect_size_name == "cohen_dz"
    assert result.effect_size is None


def test_wilcoxon_signed_rank_all_positive_exact_p_value():
    result = wilcoxon_signed_rank_test([2, 3, 4, 5], [1, 1, 1, 1])

    assert result.test_name == "wilcoxon_signed_rank"
    assert result.n == 4
    assert result.statistic == 0.0
    assert result.p_value == 0.125
    assert result.effect_size_name == "rank_biserial_correlation"
    assert result.effect_size == 1.0


def test_wilcoxon_signed_rank_ignores_zero_differences():
    result = wilcoxon_signed_rank_test([1, 2, 4], [1, 1, 1])

    assert result.n == 2
    assert result.mean_difference == 1.333333
    assert result.effect_size == 1.0


def test_compare_paired_scores_can_force_non_parametric():
    result = compare_paired_scores(
        [2.1, 2.3, 2.0],
        [1.8, 2.0, 1.9],
        assume_normal=False,
        alpha=0.1,
    )

    assert result.test_name == "wilcoxon_signed_rank"
    assert result.normality_method == "assumed_non_parametric"
    assert result.normality_alpha == 0.1
    assert "Wilcoxon" in result.test_selection_reason


def test_compare_paired_scores_can_force_t_test():
    result = compare_paired_scores(
        [2.1, 2.3, 2.0],
        [1.8, 2.0, 1.9],
        assume_normal=True,
    )

    assert result.test_name == "paired_t_test"
    assert result.normality_method == "assumed_normal"
    assert "paired t-test" in result.test_selection_reason


def test_compare_paired_scores_validates_lengths():
    with pytest.raises(ValueError):
        compare_paired_scores([1.0], [1.0, 2.0])
