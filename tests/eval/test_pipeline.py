import json
import sys
from pathlib import Path
from uuid import uuid4

from mascan.eval.pipeline import (
    bind_manifest_path,
    build_post_human_commands,
    build_pre_human_commands,
    command_outputs_satisfied,
    run_pipeline_commands,
    PipelineCommand,
)
from mascan.eval.readiness import GoldExperimentManifest


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
            "rater_ids": ["rater_1", "rater_2"],
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


def test_build_pre_human_commands_orders_pipeline_steps():
    commands = build_pre_human_commands(
        _manifest(),
        python_executable="python",
        judge_model="judge-model",
        reviewer_out_dir="eval_results/human_reviewers",
        trace_csv_file="eval_results/case_trace.csv",
    )

    names = [command.name for command in commands]

    assert names[:3] == [
        "collect_mascan",
        "collect_zero_shot_same_model",
        "collect_frontier_model",
    ]
    assert "merge_responses" in names
    assert names.index("merge_responses") < names.index("judge_responses")
    assert names.index("judge_responses") < names.index("apply_pricing")
    assert "export_human_reviewer_files" in names
    assert names[-1] == "render_report_pre_human"
    judge = next(command for command in commands if command.name == "judge_responses")
    assert judge.argv[-2:] == ["--model", "judge-model"]
    trace = next(command for command in commands if command.name == "export_case_trace")
    assert "eval_results/case_trace.csv" in trace.argv
    comparison = next(command for command in commands if command.name.startswith("compare_"))
    assert "--normality-alpha" in comparison.argv
    assert comparison.argv[comparison.argv.index("--normality-alpha") + 1] == "0.05"


def test_run_gold_pre_human_dry_run_outputs_json(mocker, capsys):
    test_dir = Path("tmp") / "test_pipeline" / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = test_dir / "manifest.json"
    manifest_path.write_text(_manifest().model_dump_json(), encoding="utf-8")
    mocker.patch.object(
        sys,
        "argv",
        ["run_gold_pre_human.py", "--manifest", str(manifest_path)],
    )

    from scripts import run_gold_pre_human

    assert run_gold_pre_human.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload[0]["name"] == "collect_mascan"
    assert any(command["name"] == "judge_responses" for command in payload)


def test_run_gold_pre_human_execute_blocks_when_preflight_fails(mocker, capsys):
    test_dir = Path("tmp") / "test_pipeline" / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = test_dir / "manifest.json"
    preflight_json = test_dir / "preflight.json"
    preflight_md = test_dir / "preflight.md"
    manifest_path.write_text(_manifest().model_dump_json(), encoding="utf-8")
    mocker.patch.object(
        sys,
        "argv",
        [
            "run_gold_pre_human.py",
            "--manifest",
            str(manifest_path),
            "--execute",
            "--preflight-out",
            str(preflight_json),
            "--preflight-markdown-out",
            str(preflight_md),
        ],
    )

    from scripts import run_gold_pre_human
    from mascan.eval.preflight import GoldPreflightReport, PreflightIssue

    mocker.patch.object(
        run_gold_pre_human,
        "run_gold_preflight",
        return_value=GoldPreflightReport(
            is_ready=False,
            phase="pre_human",
            errors=1,
            warnings=0,
            issues=[
                PreflightIssue(
                    severity="error",
                    item="dependency:langchain",
                    message="missing",
                )
            ],
        ),
    )
    run_mock = mocker.patch.object(run_gold_pre_human, "run_pipeline_commands")

    assert run_gold_pre_human.main() == 1
    assert "dependency:langchain" in capsys.readouterr().out
    assert json.loads(preflight_json.read_text(encoding="utf-8"))["is_ready"] is False
    assert "# Gold Experiment Preflight Report" in preflight_md.read_text(encoding="utf-8")
    run_mock.assert_not_called()


def test_run_gold_responses_error_config_preserves_direct_llm_controls():
    from scripts.run_gold_responses import _error_generation_config

    config = _error_generation_config("zero_shot_same_model", "gpt-4o-mini")

    assert config["runner"] == "direct_llm"
    assert config["model"] == "gpt-4o-mini"
    assert config["temperature"] == 0
    assert config["max_tokens"] == 4000
    assert config["prompt_contract"] == "gold_standard_pestel_v1"


def test_build_post_human_commands_imports_ratings_then_postprocesses():
    commands = bind_manifest_path(
        build_post_human_commands(
            _manifest(),
            ratings_csv_files=["rater_1.csv", "rater_2.csv"],
            readiness_out="eval_results/readiness_report.json",
            methodology_out="eval_results/gold_methodology_appendix.md",
            python_executable="python",
        ),
        manifest_path="manifest.json",
    )

    names = [command.name for command in commands]

    assert names == [
        "import_human_ratings_csv",
        "postprocess_gold_experiment",
        "render_methodology_appendix",
    ]
    assert commands[0].argv[:4] == [
        "python",
        "scripts\\import_human_ratings_csv.py",
        "--csv",
        "rater_1.csv",
    ]
    assert "manifest.json" in commands[1].argv
    assert "eval_results/human_irr.json" in commands[1].outputs


def test_run_gold_post_human_dry_run_outputs_json(mocker, capsys):
    test_dir = Path("tmp") / "test_pipeline" / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = test_dir / "manifest.json"
    manifest_path.write_text(_manifest().model_dump_json(), encoding="utf-8")
    mocker.patch.object(
        sys,
        "argv",
        [
            "run_gold_post_human.py",
            "--manifest",
            str(manifest_path),
            "--ratings-csv",
            "r1.csv",
            "r2.csv",
        ],
    )

    from scripts import run_gold_post_human

    assert run_gold_post_human.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload[0]["name"] == "import_human_ratings_csv"
    assert payload[-1]["name"] == "render_methodology_appendix"


def test_run_gold_post_human_execute_runs_when_preflight_passes(mocker):
    test_dir = Path("tmp") / "test_pipeline" / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = test_dir / "manifest.json"
    preflight_json = test_dir / "post_preflight.json"
    preflight_md = test_dir / "post_preflight.md"
    manifest_path.write_text(_manifest().model_dump_json(), encoding="utf-8")
    mocker.patch.object(
        sys,
        "argv",
        [
            "run_gold_post_human.py",
            "--manifest",
            str(manifest_path),
            "--ratings-csv",
            "r1.csv",
            "--execute",
            "--skip-existing",
            "--preflight-out",
            str(preflight_json),
            "--preflight-markdown-out",
            str(preflight_md),
        ],
    )

    from scripts import run_gold_post_human
    from mascan.eval.preflight import GoldPreflightReport

    mocker.patch.object(
        run_gold_post_human,
        "run_gold_preflight",
        return_value=GoldPreflightReport(
            is_ready=True,
            phase="post_human",
            errors=0,
            warnings=0,
            issues=[],
        ),
    )
    run_mock = mocker.patch.object(run_gold_post_human, "run_pipeline_commands")

    assert run_gold_post_human.main() == 0
    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["skip_existing"] is True
    assert json.loads(preflight_json.read_text(encoding="utf-8"))["is_ready"] is True
    assert "No preflight issues found." in preflight_md.read_text(encoding="utf-8")


def test_command_outputs_satisfied_requires_nonempty_files_and_dirs():
    test_dir = Path("tmp") / "test_pipeline" / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    file_path = test_dir / "out.json"
    dir_path = test_dir / "reviewers"
    command = PipelineCommand(
        name="step",
        argv=["python", "-c", "pass"],
        outputs=[str(file_path), str(dir_path)],
    )

    assert command_outputs_satisfied(command) is False
    file_path.write_text("{}", encoding="utf-8")
    dir_path.mkdir()
    assert command_outputs_satisfied(command) is False
    (dir_path / "rater.csv").write_text("x", encoding="utf-8")

    assert command_outputs_satisfied(command) is True


def test_run_pipeline_commands_skip_existing(mocker):
    test_dir = Path("tmp") / "test_pipeline" / uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    existing = test_dir / "existing.json"
    missing = test_dir / "missing.json"
    existing.write_text("{}", encoding="utf-8")
    first = PipelineCommand(
        name="first",
        argv=["python", "-c", "first"],
        outputs=[str(existing)],
    )
    second = PipelineCommand(
        name="second",
        argv=["python", "-c", "second"],
        outputs=[str(missing)],
    )
    run_mock = mocker.patch("mascan.eval.pipeline.subprocess.run")

    run_pipeline_commands([first, second], skip_existing=True)

    run_mock.assert_called_once()
    assert run_mock.call_args.args[0] == second.argv
