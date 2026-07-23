from pathlib import Path
from uuid import uuid4

from mascan.eval.pipeline import (
    PipelineCommand,
    command_outputs_satisfied,
    run_pipeline_commands,
)


def test_run_gold_responses_error_config_preserves_direct_llm_controls():
    from scripts.run_gold_responses import _error_generation_config

    config = _error_generation_config("zero_shot_same_model", "gpt-4o-mini")

    assert config["runner"] == "direct_llm"
    assert config["model"] == "gpt-4o-mini"
    assert config["temperature"] == 0
    assert config["max_tokens"] == 4000
    assert config["prompt_contract"] == "gold_standard_pestel_v1"


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
