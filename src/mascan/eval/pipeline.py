"""Command planning/execution helpers for gold-standard experiment workflows."""

import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from mascan.eval.readiness import GoldExperimentManifest


class PipelineCommand(BaseModel):
    name: str
    argv: list[str]
    outputs: list[str] = Field(default_factory=list)


def build_gold_eval_commands(
    manifest: GoldExperimentManifest,
    *,
    python_executable: str = sys.executable,
    scripts_dir: str | Path = "scripts",
    judge_model: str | None = None,
    allow_missing_price: bool = False,
    trace_csv_file: str | None = None,
) -> list[PipelineCommand]:
    """Build the ordered command list for the gold-standard evaluation pipeline."""
    scripts = Path(scripts_dir)
    commands: list[PipelineCommand] = []

    for system in manifest.systems:
        commands.append(
            PipelineCommand(
                name=f"collect_{system.system_id}",
                argv=[
                    python_executable,
                    str(scripts / "run_gold_responses.py"),
                    "--system",
                    system.system_id,
                    "--model",
                    system.model,
                    "--gold-standard",
                    manifest.gold_standard_file,
                    "--out",
                    system.response_file,
                ],
                outputs=[system.response_file],
            )
        )

    if manifest.merged_responses_file:
        commands.append(
            PipelineCommand(
                name="merge_responses",
                argv=[
                    python_executable,
                    str(scripts / "merge_gold_responses.py"),
                    "--responses",
                    *(system.response_file for system in manifest.systems),
                    "--out",
                    manifest.merged_responses_file,
                ],
                outputs=[manifest.merged_responses_file],
            )
        )

    if manifest.judged_file:
        if not manifest.merged_responses_file:
            raise ValueError("judged_file requires merged_responses_file")
        judge_argv = [
            python_executable,
            str(scripts / "run_gold_judge_batch.py"),
            "--responses",
            manifest.merged_responses_file,
            "--gold-standard",
            manifest.gold_standard_file,
            "--out",
            manifest.judged_file,
        ]
        if judge_model:
            judge_argv += ["--model", judge_model]
        commands.append(
            PipelineCommand(
                name="judge_responses",
                argv=judge_argv,
                outputs=[manifest.judged_file],
            )
        )

    analysis_judged_file = manifest.priced_judged_file or manifest.judged_file
    if manifest.pricing_file and manifest.priced_judged_file:
        if not manifest.judged_file:
            raise ValueError("priced_judged_file requires judged_file")
        pricing_argv = [
            python_executable,
            str(scripts / "apply_gold_pricing.py"),
            "--judged",
            manifest.judged_file,
            "--pricing",
            manifest.pricing_file,
            "--out",
            manifest.priced_judged_file,
        ]
        if allow_missing_price:
            pricing_argv.append("--allow-missing-price")
        commands.append(
            PipelineCommand(
                name="apply_pricing",
                argv=pricing_argv,
                outputs=[manifest.priced_judged_file],
            )
        )

    if manifest.system_summary_file and analysis_judged_file:
        commands.append(
            PipelineCommand(
                name="summarize_systems",
                argv=[
                    python_executable,
                    str(scripts / "summarize_gold_experiment.py"),
                    "--judged",
                    analysis_judged_file,
                    "--out",
                    manifest.system_summary_file,
                ],
                outputs=[manifest.system_summary_file],
            )
        )

    if manifest.case_trace_file and analysis_judged_file:
        trace_argv = [
            python_executable,
            str(scripts / "export_gold_trace.py"),
            "--judged",
            analysis_judged_file,
            "--json-out",
            manifest.case_trace_file,
        ]
        outputs = [manifest.case_trace_file]
        if trace_csv_file:
            trace_argv += ["--csv-out", trace_csv_file]
            outputs.append(trace_csv_file)
        commands.append(
            PipelineCommand(
                name="export_case_trace",
                argv=trace_argv,
                outputs=outputs,
            )
        )

    for comparison in manifest.comparisons:
        if analysis_judged_file:
            commands.append(
                PipelineCommand(
                    name=f"compare_{comparison.treatment_system}_vs_{comparison.control_system}",
                    argv=[
                        python_executable,
                        str(scripts / "compare_gold_systems.py"),
                        "--judged",
                        analysis_judged_file,
                        "--treatment-system",
                        comparison.treatment_system,
                        "--control-system",
                        comparison.control_system,
                        "--metric",
                        comparison.metric,
                        "--normality",
                        _normality_arg(comparison.assume_normal),
                        "--normality-alpha",
                        str(comparison.normality_alpha),
                        "--alternative",
                        comparison.alternative,
                        "--out",
                        comparison.file,
                    ],
                    outputs=[comparison.file],
                )
            )

    if manifest.final_report_file and manifest.system_summary_file:
        report_argv = [
            python_executable,
            str(scripts / "render_gold_report.py"),
            "--summary",
            manifest.system_summary_file,
            "--out",
            manifest.final_report_file,
        ]
        for comparison in manifest.comparisons:
            report_argv += ["--comparison", comparison.file]
        commands.append(
            PipelineCommand(
                name="render_report",
                argv=report_argv,
                outputs=[manifest.final_report_file],
            )
        )

    return commands


def bind_manifest_path(
    commands: list[PipelineCommand],
    *,
    manifest_path: str,
) -> list[PipelineCommand]:
    """Replace manifest placeholders after loading a manifest."""
    return [
        command.model_copy(
            update={
                "argv": [
                    manifest_path if arg == "__MANIFEST_PATH__" else arg
                    for arg in command.argv
                ]
            }
        )
        for command in commands
    ]


def run_pipeline_commands(
    commands: list[PipelineCommand],
    *,
    cwd: str | Path = ".",
    skip_existing: bool = False,
) -> None:
    """Run planned commands sequentially, failing on the first error."""
    base = Path(cwd)
    _ensure_output_dirs(commands, base=base)
    env = _child_env(base)
    for command in commands:
        if skip_existing and command_outputs_satisfied(command, base=base):
            print(f"Skipping {command.name}: outputs already exist.")
            continue
        subprocess.run(command.argv, cwd=base, env=env, check=True)


def command_outputs_satisfied(
    command: PipelineCommand,
    *,
    base: str | Path = ".",
) -> bool:
    """Return True when every declared command output already exists."""
    if not command.outputs:
        return False
    root = Path(base)
    return all(_output_satisfied(output, base=root) for output in command.outputs)


def _ensure_output_dirs(commands: list[PipelineCommand], *, base: Path) -> None:
    for command in commands:
        for output in command.outputs:
            output_path = Path(output)
            if output_path.suffix:
                resolved = output_path if output_path.is_absolute() else base / output_path
                resolved.parent.mkdir(parents=True, exist_ok=True)


def _output_satisfied(output: str, *, base: Path) -> bool:
    output_path = Path(output)
    resolved = output_path if output_path.is_absolute() else base / output_path
    if resolved.is_file():
        return resolved.stat().st_size > 0
    if resolved.is_dir():
        return any(resolved.iterdir())
    return False


def _child_env(base: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src_path = str((base / "src").resolve())
    root_path = str(base.resolve())
    parts = [src_path, root_path]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _normality_arg(value: bool | None) -> str:
    if value is None:
        return "auto"
    return "true" if value else "false"
