"""One-command offline post-processing for gold-standard experiment artifacts."""

import json
from pathlib import Path

from pydantic import BaseModel

from mascan.eval.costing import PricingTable, apply_pricing_to_judged
from mascan.eval.gold_analysis import (
    SystemComparison,
    case_trace_records,
    compare_systems,
    summarize_systems,
)
from mascan.eval.gold_experiment import JudgedModelResponse, SystemMetricSummary
from mascan.eval.gold_report import render_gold_experiment_report
from mascan.eval.readiness import (
    GoldExperimentManifest,
    ReadinessReport,
    validate_experiment_manifest,
)


def run_gold_postprocess(
    manifest: GoldExperimentManifest,
    *,
    base_dir: str | Path = ".",
) -> ReadinessReport:
    """Regenerate derived offline artifacts and return a readiness report."""
    base = Path(base_dir)
    judged = _load_json_list(_required(manifest.judged_file, "judged_file"), JudgedModelResponse, base)

    analysis_records = judged
    if manifest.pricing_file and manifest.priced_judged_file:
        pricing = _load_model(manifest.pricing_file, PricingTable, base)
        analysis_records = apply_pricing_to_judged(judged, pricing)
        _write_json_list(manifest.priced_judged_file, analysis_records, base)

    summaries: list[SystemMetricSummary] = []
    if manifest.system_summary_file:
        summaries = summarize_systems(analysis_records)
        _write_json_list(manifest.system_summary_file, summaries, base)

    if manifest.case_trace_file:
        _write_json_list(
            manifest.case_trace_file,
            case_trace_records(analysis_records),
            base,
        )

    comparisons: list[SystemComparison] = []
    for comparison_manifest in manifest.comparisons:
        comparison = compare_systems(
            analysis_records,
            treatment_system=comparison_manifest.treatment_system,
            control_system=comparison_manifest.control_system,
            metric=comparison_manifest.metric,
            assume_normal=comparison_manifest.assume_normal,
            normality_alpha=comparison_manifest.normality_alpha,
            alternative=comparison_manifest.alternative,
        )
        comparisons.append(comparison)
        _write_json_model(comparison_manifest.file, comparison, base)

    if manifest.final_report_file:
        if not summaries and manifest.system_summary_file:
            summaries = _load_json_list(
                manifest.system_summary_file,
                SystemMetricSummary,
                base,
            )
        _write_text(
            manifest.final_report_file,
            render_gold_experiment_report(
                summaries,
                comparisons=comparisons,
            ),
            base,
        )

    return validate_experiment_manifest(manifest, base_dir=base)


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"Manifest is missing required field for postprocess: {name}")
    return value


def _load_model(path: str, model_type: type[BaseModel], base: Path):
    return model_type.model_validate_json(_resolve(base, path).read_text(encoding="utf-8"))


def _load_json_list(path: str, model_type: type[BaseModel], base: Path):
    payload = json.loads(_resolve(base, path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [model_type.model_validate(item) for item in payload]


def _write_json_model(path: str, model: BaseModel, base: Path) -> None:
    resolved = _resolve(base, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_json_list(path: str, models: list[BaseModel], base: Path) -> None:
    resolved = _resolve(base, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps([model.model_dump(mode="json") for model in models], indent=2),
        encoding="utf-8",
    )


def _write_text(path: str, text: str, base: Path) -> None:
    resolved = _resolve(base, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def _resolve(base: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate
