"""Render a compact Markdown report for the gold-standard experiment."""

from collections.abc import Sequence

from mascan.eval.gold_analysis import SystemComparison
from mascan.eval.gold_experiment import SystemMetricSummary


def render_gold_experiment_report(
    summaries: Sequence[SystemMetricSummary],
    *,
    comparisons: Sequence[SystemComparison] = (),
) -> str:
    lines = [
        "# Gold-Standard PESTEL Evaluation Report",
        "",
        "## System Summary",
        "",
        "| System | n | Depth | Categorization | Combined | Quality points | Missing gold claims | Unsupported/wrong claims | Tokens | Cost USD | Quality / 1k tokens | Quality / USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.system_id} | "
            f"{summary.n} | "
            f"{_fmt(summary.mean_analytical_depth)} | "
            f"{_fmt(summary.mean_categorization_accuracy)} | "
            f"{_fmt(summary.mean_combined_quality)} | "
            f"{_fmt(summary.total_quality_points)} | "
            f"{summary.total_missing_gold_claims if summary.total_missing_gold_claims is not None else '-'} | "
            f"{summary.total_unsupported_or_wrong_claims if summary.total_unsupported_or_wrong_claims is not None else '-'} | "
            f"{summary.total_tokens if summary.total_tokens is not None else '-'} | "
            f"{_fmt(summary.total_cost_usd, digits=6)} | "
            f"{_fmt(summary.quality_per_1k_tokens, digits=6)} | "
            f"{_fmt(summary.quality_per_usd, digits=6)} |"
        )

    if comparisons:
        lines += [
            "",
            "## Paired Significance Tests",
            "",
            "| Treatment | Control | Metric | Pairs | Test | Statistic | p-value | Mean diff | Effect size | Normality |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
        for comparison in comparisons:
            result = comparison.result
            normality = result.normality_method or "-"
            if result.normality_alpha is not None:
                normality = f"{normality} alpha={result.normality_alpha:.4f}"
            if result.normality_p_value is not None:
                normality = f"{normality} p={result.normality_p_value:.4f}"
            effect = "-"
            if result.effect_size is not None:
                effect_name = result.effect_size_name or "effect"
                effect = f"{effect_name}={result.effect_size:.4f}"
            lines.append(
                "| "
                f"{comparison.treatment_system} | "
                f"{comparison.control_system} | "
                f"{comparison.metric} | "
                f"{comparison.paired_cases} | "
                f"{result.test_name} | "
                f"{result.statistic:.4f} | "
                f"{result.p_value:.4f} | "
                f"{result.mean_difference:.4f} | "
                f"{effect} | "
                f"{normality} |"
            )
        notes = [
            (
                comparison,
                comparison.result.test_selection_reason,
            )
            for comparison in comparisons
            if comparison.result.test_selection_reason
        ]
        if notes:
            lines += ["", "**Test Selection Notes**", ""]
            for comparison, reason in notes:
                lines.append(
                    f"- {comparison.treatment_system} vs "
                    f"{comparison.control_system} ({comparison.metric}): {reason}"
                )
        if any(comparison.paired_case_ids for comparison in comparisons):
            lines += ["", "**Paired Case Audit**", ""]
            for comparison in comparisons:
                if comparison.paired_case_ids:
                    lines.append(
                        f"- {comparison.treatment_system} vs "
                        f"{comparison.control_system} ({comparison.metric}): "
                        f"{', '.join(comparison.paired_case_ids)}"
                    )

    return "\n".join(lines) + "\n"


def _fmt(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"
