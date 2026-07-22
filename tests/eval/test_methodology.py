from mascan.eval.methodology import (
    build_methodology_checklist,
    render_methodology_appendix,
)
from mascan.eval.fingerprints import ArtifactFingerprint
from mascan.eval.readiness import (
    GoldExperimentManifest,
    ReadinessIssue,
    ReadinessReport,
)


def _manifest() -> GoldExperimentManifest:
    return GoldExperimentManifest(
        gold_standard_file="eval_papers/gold_standard_cases.json",
        expected_case_count=25,
        systems=[
            {
                "system_id": "mascan",
                "model": "gpt-4o-mini",
                "response_file": "eval_results/responses_mascan.json",
            },
            {
                "system_id": "zero_shot_same_model",
                "model": "gpt-4o-mini",
                "response_file": "eval_results/responses_zero_shot.json",
            },
            {
                "system_id": "frontier_model",
                "model": "gpt-4o",
                "response_file": "eval_results/responses_frontier.json",
            },
        ],
        merged_responses_file="eval_results/responses_all.json",
        judged_file="eval_results/judged_all.json",
        priced_judged_file="eval_results/judged_all_priced.json",
        pricing_file="eval_results/model_pricing.json",
        system_summary_file="eval_results/system_summary.json",
        case_trace_file="eval_results/case_trace.json",
        human_calibration={
            "packet_file": "eval_results/human_packet.json",
            "answer_key_file": "eval_results/human_answer_key.json",
            "ratings_template_file": "eval_results/human_ratings_template.json",
            "ratings_file": "eval_results/human_ratings.json",
            "rater_ids": ["rater_1", "rater_2", "rater_3", "rater_4", "rater_5"],
            "cases_per_rater": 5,
            "irr_file": "eval_results/human_irr.json",
            "expected_case_count": 25,
        },
        comparisons=[
            {
                "treatment_system": "mascan",
                "control_system": "zero_shot_same_model",
                "metric": "combined_quality",
                "file": "eval_results/mascan_vs_zero_shot.json",
            }
        ],
        final_report_file="eval_results/gold_experiment_report.md",
    )


def test_methodology_checklist_maps_readiness_issues_to_steps():
    readiness = ReadinessReport(
        is_ready=False,
        errors=2,
        warnings=0,
        issues=[
            ReadinessIssue(
                severity="error",
                item="response_file:mascan",
                message="missing responses",
            ),
            ReadinessIssue(
                severity="error",
                item="case_trace_file",
                message="missing trace/cost analysis",
            ),
        ],
    )

    checklist = build_methodology_checklist(_manifest(), readiness)
    by_step = {item.step: item for item in checklist}

    assert by_step[1].status == "incomplete"  # response_file:mascan
    assert by_step[5].status == "incomplete"  # case_trace_file -> Trace And Cost Analysis
    assert by_step[2].status == "complete"


def test_render_methodology_appendix_includes_protocol_and_status():
    readiness = ReadinessReport(
        is_ready=True,
        errors=0,
        warnings=0,
        issues=[],
        fingerprints=[
            ArtifactFingerprint(
                artifact="gold_standard_dataset",
                method="canonical_pydantic_json_sha256",
                sha256="abc123",
            )
        ],
    )

    markdown = render_methodology_appendix(_manifest(), readiness=readiness)

    assert "Gold-Standard PESTEL Evaluation Methodology Appendix" in markdown
    assert "Analytical Depth" in markdown
    assert "Readiness Checklist" in markdown
    assert "eval_results/responses_mascan.json" in markdown
    assert "Reproducibility Fingerprints" in markdown
    assert "`abc123`" in markdown
    assert "- Ready: true" in markdown
