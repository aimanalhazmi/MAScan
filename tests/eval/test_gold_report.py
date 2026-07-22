from mascan.eval.gold_analysis import SystemComparison
from mascan.eval.gold_experiment import MetricPairRecord, SystemMetricSummary
from mascan.eval.gold_report import render_gold_experiment_report
from mascan.eval.stats import PairedTestResult


def test_render_gold_experiment_report_includes_core_sections():
    report = render_gold_experiment_report(
        [
            SystemMetricSummary(
                system_id="mascan",
                n=25,
                mean_analytical_depth=2.4,
                mean_categorization_accuracy=0.8,
                mean_combined_quality=0.8,
                total_quality_points=20.0,
                total_missing_gold_claims=4,
                total_unsupported_or_wrong_claims=2,
                total_tokens=1000,
                total_cost_usd=0.01,
                quality_per_1k_tokens=20.0,
                quality_per_usd=2000.0,
            )
        ],
        comparisons=[
            SystemComparison(
                treatment_system="mascan",
                control_system="baseline",
                metric="combined_quality",
                paired_cases=25,
                paired_case_ids=["case_a"],
                treatment_scores=[0.8],
                control_scores=[0.7],
                score_pairs=[
                    MetricPairRecord(
                        case_id="case_a",
                        treatment_score=0.8,
                        control_score=0.7,
                        difference=0.1,
                    )
                ],
                result=PairedTestResult(
                    test_name="wilcoxon_signed_rank",
                    alternative="two-sided",
                    n=25,
                    statistic=10.0,
                    p_value=0.03,
                    mean_difference=0.1,
                    effect_size_name="rank_biserial_correlation",
                    effect_size=0.7,
                    normality_method="assumed_non_parametric",
                    normality_alpha=0.05,
                    test_selection_reason="Normality not assumed by caller; selected Wilcoxon signed-rank.",
                ),
            )
        ],
    )

    assert "# Gold-Standard PESTEL Evaluation Report" in report
    assert "mascan" in report
    assert "Paired Significance Tests" in report
    assert "Test Selection Notes" in report
    assert "Paired Case Audit" in report
    assert "case_a" in report
    assert "Quality / USD" in report
    assert "Effect size" in report
    assert "rank_biserial_correlation=0.7000" in report
    assert "assumed_non_parametric alpha=0.0500" in report
    assert "Missing gold claims" in report
    assert "Unsupported/wrong claims" in report
