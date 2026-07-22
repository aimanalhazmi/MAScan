"""Readiness checks for a full gold-standard PESTEL experiment run."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from mascan.eval.costing import PricingTable, validate_pricing_table
from mascan.eval.fingerprints import ArtifactFingerprint, file_sha256, model_sha256
from mascan.eval.gold_analysis import CaseTraceRecord, SystemComparison
from mascan.eval.gold_experiment import (
    JudgedModelResponse,
    ModelResponseRecord,
    SystemMetricSummary,
    prompt_sha256,
)
from mascan.eval.gold_judge import (
    GoldJudgeResult,
    gold_judge_prompt_sha256,
    gold_judge_schema_sha256,
)
from mascan.eval.gold_standard import GoldStandardDataset
from mascan.eval.stats import Alternative

Severity = Literal["error", "warning"]


class ExperimentSystemManifest(BaseModel):
    system_id: str
    model: str
    response_file: str


class ComparisonManifest(BaseModel):
    treatment_system: str
    control_system: str
    metric: str = "combined_quality"
    file: str
    assume_normal: bool | None = None
    normality_alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    alternative: Alternative = "two-sided"


class GoldExperimentManifest(BaseModel):
    gold_standard_file: str = "eval_papers/gold_standard_cases.json"
    expected_case_count: int = 25
    systems: list[ExperimentSystemManifest] = Field(default_factory=list)
    merged_responses_file: str | None = None
    judged_file: str | None = None
    priced_judged_file: str | None = None
    pricing_file: str | None = None
    system_summary_file: str | None = None
    case_trace_file: str | None = None
    comparisons: list[ComparisonManifest] = Field(default_factory=list)
    final_report_file: str | None = None


class ReadinessIssue(BaseModel):
    severity: Severity
    item: str
    message: str


class ReadinessReport(BaseModel):
    is_ready: bool
    errors: int
    warnings: int
    issues: list[ReadinessIssue] = Field(default_factory=list)
    fingerprints: list[ArtifactFingerprint] = Field(default_factory=list)


def load_experiment_manifest(path: str | Path) -> GoldExperimentManifest:
    return GoldExperimentManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def validate_experiment_manifest(
    manifest: GoldExperimentManifest,
    *,
    base_dir: str | Path = ".",
) -> ReadinessReport:
    base = Path(base_dir)
    issues: list[ReadinessIssue] = []
    fingerprints = [
        ArtifactFingerprint(
            artifact="experiment_manifest",
            sha256=model_sha256(manifest),
            method="canonical_pydantic_json_sha256",
        ),
        ArtifactFingerprint(
            artifact="judge_system_prompt",
            sha256=gold_judge_prompt_sha256(),
            method="utf8_text_sha256",
        ),
        ArtifactFingerprint(
            artifact="judge_output_schema",
            sha256=gold_judge_schema_sha256(),
            method="canonical_json_sha256",
        ),
    ]

    gold = _load_model(
        manifest.gold_standard_file,
        GoldStandardDataset,
        base,
        issues,
        "gold_standard_file",
    )
    case_ids: set[str] = set()
    prompt_hashes: dict[str, str] = {}
    category_targets: dict[str, list[tuple[str, str]]] = {}
    expected_systems = {system.system_id for system in manifest.systems}
    if gold is not None:
        fingerprints.extend(
            [
                ArtifactFingerprint(
                    artifact=manifest.gold_standard_file,
                    sha256=file_sha256(_resolve(base, manifest.gold_standard_file)),
                    method="file_bytes_sha256",
                ),
                ArtifactFingerprint(
                    artifact="gold_standard_dataset",
                    sha256=model_sha256(gold),
                    method="canonical_pydantic_json_sha256",
                ),
            ]
        )
        case_ids = {case.case_id for case in gold.cases}
        prompt_hashes = {case.case_id: prompt_sha256(case.prompt) for case in gold.cases}
        category_targets = {
            case.case_id: [
                (target.factor, target.correct_category)
                for target in case.category_targets
            ]
            for case in gold.cases
        }
        _expect(
            len(gold.cases) == manifest.expected_case_count,
            issues,
            "gold_standard_file",
            f"Expected {manifest.expected_case_count} cases, found {len(gold.cases)}.",
        )
        for case in gold.cases:
            _expect(
                _resolve(base, case.source_pdf).exists(),
                issues,
                f"source_pdf:{case.case_id}",
                f"Missing source PDF: {case.source_pdf}",
            )
            _expect(
                bool(case.gold_claims and case.category_targets),
                issues,
                f"gold_targets:{case.case_id}",
                "Case must include gold_claims and category_targets.",
            )

    system_records: list[ModelResponseRecord] = []
    for system in manifest.systems:
        records = _load_json_list(
            system.response_file,
            ModelResponseRecord,
            base,
            issues,
            f"response_file:{system.system_id}",
        )
        if records is None:
            continue
        system_records.extend(records)
        _validate_response_records(
            records,
            system.system_id,
            system.model,
            case_ids,
            prompt_hashes,
            manifest.expected_case_count,
            issues,
            f"response_file:{system.system_id}",
        )

    if manifest.merged_responses_file:
        merged = _load_json_list(
            manifest.merged_responses_file,
            ModelResponseRecord,
            base,
            issues,
            "merged_responses_file",
        )
        if merged is not None:
            _validate_merged_responses(merged, manifest, case_ids, issues)
            _validate_record_prompt_hashes(
                merged,
                prompt_hashes,
                issues,
                "merged_responses_file",
            )

    judged_records: list[JudgedModelResponse] | None = None
    if manifest.judged_file:
        judged_records = _load_json_list(
            manifest.judged_file,
            JudgedModelResponse,
            base,
            issues,
            "judged_file",
        )
        if judged_records is not None:
            _validate_judged_records(
                judged_records,
                manifest,
                case_ids,
                prompt_hashes,
                category_targets,
                issues,
                "judged_file",
            )

    if manifest.pricing_file:
        pricing = _load_model(
            manifest.pricing_file, PricingTable, base, issues, "pricing_file"
        )
        if pricing is not None:
            _validate_pricing_table(pricing, manifest, issues)

    if manifest.priced_judged_file:
        priced = _load_json_list(
            manifest.priced_judged_file,
            JudgedModelResponse,
            base,
            issues,
            "priced_judged_file",
        )
        if priced is not None:
            judged_records = priced
            _validate_judged_records(
                priced,
                manifest,
                case_ids,
                prompt_hashes,
                category_targets,
                issues,
                "priced_judged_file",
            )
            _validate_costs(priced, issues)

    if manifest.system_summary_file:
        summaries = _load_json_list(
            manifest.system_summary_file,
            SystemMetricSummary,
            base,
            issues,
            "system_summary_file",
        )
        if summaries is not None:
            found_systems = {summary.system_id for summary in summaries}
            _expect(
                expected_systems.issubset(found_systems),
                issues,
                "system_summary_file",
                f"Missing summaries for systems: {sorted(expected_systems - found_systems)}",
            )

    if manifest.case_trace_file:
        traces = _load_json_list(
            manifest.case_trace_file,
            CaseTraceRecord,
            base,
            issues,
            "case_trace_file",
        )
        if traces is not None:
            _validate_case_traces(traces, manifest, case_ids, issues)

    for comparison in manifest.comparisons:
        parsed = _load_model(
            comparison.file,
            SystemComparison,
            base,
            issues,
            f"comparison:{comparison.treatment_system}_vs_{comparison.control_system}",
        )
        if parsed is not None:
            _expect(
                parsed.treatment_system == comparison.treatment_system
                and parsed.control_system == comparison.control_system
                and parsed.metric == comparison.metric,
                issues,
                f"comparison:{comparison.file}",
                "Comparison file metadata does not match manifest.",
            )
            _expect(
                parsed.result.normality_alpha == comparison.normality_alpha,
                issues,
                f"comparison:{comparison.file}",
                (
                    "Comparison normality_alpha does not match manifest "
                    f"({parsed.result.normality_alpha} != {comparison.normality_alpha})."
                ),
            )
            _expect(
                parsed.paired_cases == manifest.expected_case_count,
                issues,
                f"comparison:{comparison.file}",
                f"Expected {manifest.expected_case_count} paired cases, found {parsed.paired_cases}.",
            )
            _expect(
                len(parsed.paired_case_ids) == parsed.paired_cases,
                issues,
                f"comparison:{comparison.file}",
                "Comparison must include one paired_case_id per paired score.",
            )
            if parsed.paired_case_ids:
                _expect(
                    set(parsed.paired_case_ids) == case_ids,
                    issues,
                    f"comparison:{comparison.file}",
                    "Comparison paired_case_ids must match the gold-standard cases.",
                )
            _expect(
                len(parsed.score_pairs) == parsed.paired_cases,
                issues,
                f"comparison:{comparison.file}",
                "Comparison must include one score_pairs entry per paired score.",
            )
            if parsed.score_pairs and parsed.paired_case_ids:
                _expect(
                    [pair.case_id for pair in parsed.score_pairs]
                    == parsed.paired_case_ids,
                    issues,
                    f"comparison:{comparison.file}",
                    "score_pairs case IDs must match paired_case_ids in order.",
                )

    if manifest.final_report_file:
        report_path = _resolve(base, manifest.final_report_file)
        _expect(
            report_path.exists() and report_path.stat().st_size > 0,
            issues,
            "final_report_file",
            f"Missing or empty final report: {manifest.final_report_file}",
        )

    _fingerprint_manifest_files(manifest, base, fingerprints)

    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    return ReadinessReport(
        is_ready=errors == 0,
        errors=errors,
        warnings=warnings,
        issues=issues,
        fingerprints=fingerprints,
    )


def _validate_response_records(
    records: list[ModelResponseRecord],
    system_id: str,
    model: str,
    case_ids: set[str],
    prompt_hashes: dict[str, str],
    expected_case_count: int,
    issues: list[ReadinessIssue],
    item: str,
) -> None:
    found_case_ids = {record.case_id for record in records}
    _expect(len(records) == expected_case_count, issues, item, f"Expected {expected_case_count} response records, found {len(records)}.")
    _expect(all(record.system_id == system_id for record in records), issues, item, f"All records must use system_id={system_id}.")
    _expect(all(record.model == model for record in records), issues, item, f"All records must use model={model}.")
    if case_ids:
        _expect(found_case_ids == case_ids, issues, item, f"Response case IDs do not match gold standard. Missing={sorted(case_ids - found_case_ids)} extra={sorted(found_case_ids - case_ids)}")
    _validate_record_prompt_hashes(records, prompt_hashes, issues, item)
    for record in records:
        if record.error:
            _add_issue(issues, "error", item, f"{record.case_id} has error: {record.error}")
        elif not record.response_text.strip():
            _add_issue(issues, "error", item, f"{record.case_id} has empty response_text.")
        _expect(
            record.generation_config.get("prompt_contract") == "gold_standard_pestel_v1",
            issues,
            item,
            f"{record.case_id}|{record.system_id} is missing prompt_contract=gold_standard_pestel_v1 in generation_config.",
        )
        _validate_generation_config(record, model, issues, item)


def _validate_generation_config(
    record: ModelResponseRecord,
    expected_model: str,
    issues: list[ReadinessIssue],
    item: str,
) -> None:
    config = record.generation_config
    if record.system_id == "mascan":
        _expect(
            config.get("runner") == "mascan_orchestrator",
            issues,
            item,
            f"{record.case_id}|mascan must use runner=mascan_orchestrator.",
        )
        _expect(
            config.get("effective_default_model") == expected_model,
            issues,
            item,
            (
                f"{record.case_id}|mascan effective_default_model must match "
                f"manifest model={expected_model}."
            ),
        )
        agent_models = config.get("agent_models")
        _expect(
            isinstance(agent_models, dict) and bool(agent_models),
            issues,
            item,
            f"{record.case_id}|mascan must record non-empty agent_models.",
        )
        if isinstance(agent_models, dict):
            mismatched = {
                name: agent_model
                for name, agent_model in agent_models.items()
                if agent_model != expected_model
            }
            _expect(
                not mismatched,
                issues,
                item,
                (
                    f"{record.case_id}|mascan agent_models must all match "
                    f"manifest model={expected_model}; mismatched={mismatched}."
                ),
            )
        return

    _expect(
        config.get("runner") == "direct_llm",
        issues,
        item,
        f"{record.case_id}|{record.system_id} must use runner=direct_llm.",
    )
    _expect(
        config.get("model") == expected_model,
        issues,
        item,
        (
            f"{record.case_id}|{record.system_id} generation_config.model must "
            f"match manifest model={expected_model}."
        ),
    )
    _expect(
        config.get("temperature") == 0,
        issues,
        item,
        f"{record.case_id}|{record.system_id} must use temperature=0.",
    )
    _expect(
        config.get("max_tokens") == 4000,
        issues,
        item,
        f"{record.case_id}|{record.system_id} must use max_tokens=4000.",
    )


def _validate_merged_responses(
    records: list[ModelResponseRecord],
    manifest: GoldExperimentManifest,
    case_ids: set[str],
    issues: list[ReadinessIssue],
) -> None:
    expected_systems = {system.system_id for system in manifest.systems}
    expected_models = {system.system_id: system.model for system in manifest.systems}
    expected_total = manifest.expected_case_count * len(expected_systems)
    keys = {(record.case_id, record.system_id) for record in records}
    _expect(len(records) == expected_total, issues, "merged_responses_file", f"Expected {expected_total} merged records, found {len(records)}.")
    _expect(len(keys) == len(records), issues, "merged_responses_file", "Duplicate case/system records found.")
    if case_ids and expected_systems:
        missing = [
            f"{case_id}|{system_id}"
            for case_id in sorted(case_ids)
            for system_id in sorted(expected_systems)
            if (case_id, system_id) not in keys
        ]
        _expect(not missing, issues, "merged_responses_file", f"Missing merged records: {missing[:10]}")
    for record in records:
        expected_model = expected_models.get(record.system_id)
        if expected_model:
            _validate_generation_config(
                record,
                expected_model,
                issues,
                "merged_responses_file",
            )


def _validate_judged_records(
    records: list[JudgedModelResponse],
    manifest: GoldExperimentManifest,
    case_ids: set[str],
    prompt_hashes: dict[str, str],
    category_targets: dict[str, list[tuple[str, str]]],
    issues: list[ReadinessIssue],
    item: str,
) -> None:
    expected_systems = {system.system_id for system in manifest.systems}
    expected_models = {system.system_id: system.model for system in manifest.systems}
    expected_total = manifest.expected_case_count * len(expected_systems)
    keys = {(record.response.case_id, record.response.system_id) for record in records}
    _expect(len(records) == expected_total, issues, item, f"Expected {expected_total} judged records, found {len(records)}.")
    _expect(len(keys) == len(records), issues, item, "Duplicate judged case/system records found.")
    if case_ids and expected_systems:
        missing = [
            f"{case_id}|{system_id}"
            for case_id in sorted(case_ids)
            for system_id in sorted(expected_systems)
            if (case_id, system_id) not in keys
        ]
        _expect(not missing, issues, item, f"Missing judged records: {missing[:10]}")
    _validate_record_prompt_hashes(
        [record.response for record in records],
        prompt_hashes,
        issues,
        item,
    )
    for record in records:
        expected_model = expected_models.get(record.response.system_id)
        if expected_model:
            _validate_generation_config(
                record.response,
                expected_model,
                issues,
                item,
            )
        if record.error:
            _add_issue(issues, "error", item, f"{record.response.case_id}|{record.response.system_id} has judge error: {record.error}")
        elif record.judge is None:
            _add_issue(issues, "error", item, f"{record.response.case_id}|{record.response.system_id} is missing judge output.")
        else:
            _validate_judge_fingerprints(record.judge, record, issues, item)
            _validate_judged_category_targets(
                record,
                category_targets,
                issues,
                item,
            )


def _validate_judge_fingerprints(
    judge: GoldJudgeResult,
    record: JudgedModelResponse,
    issues: list[ReadinessIssue],
    item: str,
) -> None:
    label = f"{record.response.case_id}|{record.response.system_id}"
    _expect(
        judge.judge_prompt_sha256 == gold_judge_prompt_sha256(),
        issues,
        item,
        f"{label} judge_prompt_sha256 does not match the current judge prompt.",
    )
    _expect(
        judge.judge_schema_sha256 == gold_judge_schema_sha256(),
        issues,
        item,
        f"{label} judge_schema_sha256 does not match the current judge schema.",
    )


def _validate_judged_category_targets(
    record: JudgedModelResponse,
    category_targets: dict[str, list[tuple[str, str]]],
    issues: list[ReadinessIssue],
    item: str,
) -> None:
    if record.judge is None:
        return
    expected = category_targets.get(record.response.case_id)
    if expected is None:
        return
    observed = [
        (judgment.factor, judgment.expected_category)
        for judgment in record.judge.category_judgments
    ]
    _expect(
        observed == expected,
        issues,
        item,
        (
            f"{record.response.case_id}|{record.response.system_id} judge "
            "category_judgments do not preserve the gold target factors/order."
        ),
    )


def _validate_record_prompt_hashes(
    records: list[ModelResponseRecord],
    prompt_hashes: dict[str, str],
    issues: list[ReadinessIssue],
    item: str,
) -> None:
    if not prompt_hashes:
        return
    for record in records:
        expected_hash = prompt_hashes.get(record.case_id)
        if expected_hash is None:
            continue
        _expect(
            record.prompt_sha256 == expected_hash,
            issues,
            item,
            (
                f"{record.case_id}|{record.system_id} prompt_sha256 does not match "
                "the frozen gold-standard prompt."
            ),
        )


def _validate_costs(records: list[JudgedModelResponse], issues: list[ReadinessIssue]) -> None:
    for record in records:
        usage = record.response.token_usage
        if usage.prompt_tokens is not None and usage.completion_tokens is not None:
            _expect(
                usage.cost_usd is not None,
                issues,
                "priced_judged_file",
                f"{record.response.case_id}|{record.response.system_id} has token counts but no cost_usd.",
            )


def _validate_pricing_table(
    pricing: PricingTable,
    manifest: GoldExperimentManifest,
    issues: list[ReadinessIssue],
) -> None:
    validation = validate_pricing_table(
        pricing,
        expected_models=[system.model for system in manifest.systems],
    )
    for model in validation.missing_models:
        _add_issue(
            issues,
            "error",
            "pricing_file",
            f"Missing pricing entry for model: {model}",
        )
    for model in validation.duplicate_models:
        _add_issue(
            issues,
            "error",
            "pricing_file",
            f"Duplicate pricing entry for model: {model}",
        )
    for model in validation.zero_rate_models:
        _add_issue(
            issues,
            "warning",
            "pricing_file",
            f"Pricing entry for {model} is zero; cost analysis will not be meaningful.",
        )
    for field in validation.missing_metadata:
        _add_issue(
            issues,
            "warning",
            "pricing_file",
            f"Pricing table is missing citation metadata field: {field}",
        )


def _validate_case_traces(
    traces: list[CaseTraceRecord],
    manifest: GoldExperimentManifest,
    case_ids: set[str],
    issues: list[ReadinessIssue],
) -> None:
    expected_systems = {system.system_id for system in manifest.systems}
    expected_total = manifest.expected_case_count * len(expected_systems)
    keys = {(trace.case_id, trace.system_id) for trace in traces}
    _expect(
        len(traces) == expected_total,
        issues,
        "case_trace_file",
        f"Expected {expected_total} case trace rows, found {len(traces)}.",
    )
    _expect(
        len(keys) == len(traces),
        issues,
        "case_trace_file",
        "Duplicate case/system trace rows found.",
    )
    if case_ids and expected_systems:
        missing = [
            f"{case_id}|{system_id}"
            for case_id in sorted(case_ids)
            for system_id in sorted(expected_systems)
            if (case_id, system_id) not in keys
        ]
        _expect(
            not missing,
            issues,
            "case_trace_file",
            f"Missing case trace rows: {missing[:10]}",
        )
    for trace in traces:
        if trace.combined_quality is not None and trace.total_tokens:
            _expect(
                trace.quality_per_1k_tokens is not None,
                issues,
                "case_trace_file",
                f"{trace.case_id}|{trace.system_id} missing quality_per_1k_tokens.",
            )
        if trace.combined_quality is not None:
            _expect(
                trace.response_claim_count is not None,
                issues,
                "case_trace_file",
                f"{trace.case_id}|{trace.system_id} missing response_claim_count.",
            )
            _expect(
                trace.missing_gold_claim_count is not None,
                issues,
                "case_trace_file",
                f"{trace.case_id}|{trace.system_id} missing missing_gold_claim_count.",
            )
            _expect(
                trace.unsupported_or_wrong_claim_count is not None,
                issues,
                "case_trace_file",
                f"{trace.case_id}|{trace.system_id} missing unsupported_or_wrong_claim_count.",
            )
            _expect(
                trace.category_targets_evaluated is not None,
                issues,
                "case_trace_file",
                f"{trace.case_id}|{trace.system_id} missing category_targets_evaluated.",
            )
            _expect(
                trace.category_targets_correct is not None,
                issues,
                "case_trace_file",
                f"{trace.case_id}|{trace.system_id} missing category_targets_correct.",
            )


def _load_model[M: BaseModel](
    path: str,
    model_type: type[M],
    base: Path,
    issues: list[ReadinessIssue],
    item: str,
) -> M | None:
    resolved = _resolve(base, path)
    if not resolved.exists():
        _add_issue(issues, "error", item, f"Missing file: {path}")
        return None
    try:
        return model_type.model_validate_json(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        _add_issue(issues, "error", item, f"Could not parse {path}: {exc}")
        return None


def _load_json_list[M: BaseModel](
    path: str,
    model_type: type[M],
    base: Path,
    issues: list[ReadinessIssue],
    item: str,
) -> list[M] | None:
    resolved = _resolve(base, path)
    if not resolved.exists():
        _add_issue(issues, "error", item, f"Missing file: {path}")
        return None
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON list")
        return [model_type.model_validate(entry) for entry in payload]
    except Exception as exc:
        _add_issue(issues, "error", item, f"Could not parse {path}: {exc}")
        return None


def _fingerprint_manifest_files(
    manifest: GoldExperimentManifest,
    base: Path,
    fingerprints: list[ArtifactFingerprint],
) -> None:
    """Add byte-level fingerprints for every existing file named by the manifest."""
    paths: list[str] = [
        manifest.gold_standard_file,
        *(system.response_file for system in manifest.systems),
    ]
    paths.extend(
        path
        for path in [
            manifest.merged_responses_file,
            manifest.judged_file,
            manifest.priced_judged_file,
            manifest.pricing_file,
            manifest.system_summary_file,
            manifest.case_trace_file,
            manifest.final_report_file,
        ]
        if path
    )
    paths.extend(comparison.file for comparison in manifest.comparisons)

    seen = {(fingerprint.artifact, fingerprint.method) for fingerprint in fingerprints}
    for path in paths:
        resolved = _resolve(base, path)
        key = (path, "file_bytes_sha256")
        if key in seen or not resolved.is_file():
            continue
        fingerprints.append(
            ArtifactFingerprint(
                artifact=path,
                sha256=file_sha256(resolved),
                method="file_bytes_sha256",
            )
        )
        seen.add(key)


def _resolve(base: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _expect(
    condition: bool,
    issues: list[ReadinessIssue],
    item: str,
    message: str,
    *,
    severity: Severity = "error",
) -> None:
    if not condition:
        _add_issue(issues, severity, item, message)


def _add_issue(
    issues: list[ReadinessIssue],
    severity: Severity,
    item: str,
    message: str,
) -> None:
    issues.append(ReadinessIssue(severity=severity, item=item, message=message))
