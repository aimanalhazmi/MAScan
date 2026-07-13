"""Preflight checks before running the paid gold-standard experiment pipeline."""

import csv
import importlib.util
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from mascan.eval.costing import PricingTable, validate_pricing_table
from mascan.eval.gold_standard import GoldStandardDataset
from mascan.eval.human_calibration import HumanCalibrationPacket
from mascan.eval.human_ratings import (
    HumanRatingsFile,
    human_ratings_from_csv_rows,
    validate_complete_human_ratings,
)
from mascan.eval.readiness import GoldExperimentManifest

PreflightPhase = Literal["pre_human", "post_human"]
Severity = Literal["error", "warning"]


class PreflightIssue(BaseModel):
    severity: Severity
    item: str
    message: str


class GoldPreflightReport(BaseModel):
    is_ready: bool
    phase: PreflightPhase
    errors: int
    warnings: int
    issues: list[PreflightIssue] = Field(default_factory=list)


def render_gold_preflight_markdown(report: GoldPreflightReport) -> str:
    """Render preflight results as a human-readable action checklist."""
    status = "ready" if report.is_ready else "blocked"
    lines = [
        "# Gold Experiment Preflight Report",
        "",
        f"- Phase: `{report.phase}`",
        f"- Status: {status}",
        f"- Errors: {report.errors}",
        f"- Warnings: {report.warnings}",
        "",
    ]
    if not report.issues:
        lines.append("No preflight issues found.")
        return "\n".join(lines).rstrip() + "\n"

    lines += [
        "| Severity | Item | Issue | Action |",
        "|---|---|---|---|",
    ]
    for issue in report.issues:
        lines.append(
            "| "
            f"{issue.severity} | "
            f"`{issue.item}` | "
            f"{_escape_table_cell(issue.message)} | "
            f"{_escape_table_cell(_preflight_action(issue))} |"
        )
    return "\n".join(lines).rstrip() + "\n"


PRE_HUMAN_REQUIRED_MODULES = (
    "dotenv",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "openai",
    "pydantic",
    "pydantic_settings",
    "yaml",
)

MASCAN_TOOL_MODULES = (
    "firecrawl",
    "newsdataapi",
    "yfinance",
)

POST_HUMAN_REQUIRED_MODULES = (
    "pydantic",
)


def run_gold_preflight(
    manifest: GoldExperimentManifest,
    *,
    base_dir: str | Path = ".",
    phase: PreflightPhase = "pre_human",
    ratings_csv_files: list[str] | None = None,
) -> GoldPreflightReport:
    """Validate inputs/environment before running a real experiment phase."""
    base = Path(base_dir)
    issues: list[PreflightIssue] = []

    _check_python_version(
        issues,
        severity="error" if phase == "pre_human" else "warning",
    )
    _check_modules(
        PRE_HUMAN_REQUIRED_MODULES if phase == "pre_human" else POST_HUMAN_REQUIRED_MODULES,
        issues,
    )
    if phase == "pre_human":
        _check_modules(MASCAN_TOOL_MODULES, issues, severity="warning")
        _check_openai_key(base, issues)
        _check_mascan_tool_credentials(base, issues)
        _check_gold_standard(manifest, base, issues)
        _check_pricing_table(manifest, base, issues)
    else:
        _check_post_human_inputs(manifest, base, ratings_csv_files, issues)

    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    return GoldPreflightReport(
        is_ready=errors == 0,
        phase=phase,
        errors=errors,
        warnings=warnings,
        issues=issues,
    )


def _check_python_version(
    issues: list[PreflightIssue],
    *,
    severity: Severity,
) -> None:
    version = sys.version_info
    if version.major != 3 or version.minor != 12:
        _add_issue(
            issues,
            severity,
            "python_version",
            (
                "pyproject.toml declares Python >=3.12,<3.13; current runtime is "
                f"{version.major}.{version.minor}.{version.micro}. "
                "Use a Python 3.12 environment for the paid pre-human model "
                "collection and judging phase."
            ),
        )


def _preflight_action(issue: PreflightIssue) -> str:
    item = issue.item
    message = issue.message
    if item == "python_version":
        return (
            "Run the paid pre-human phase from an activated Python 3.12 "
            "environment, then rerun preflight."
        )
    if item.startswith("dependency:"):
        module_name = item.split(":", 1)[1]
        return (
            f"Install or activate the project environment that provides "
            f"`{module_name}`; for a fresh environment, install the project "
            "dependencies from `pyproject.toml`."
        )
    if item == "env:OPENAI_API_KEY":
        return "Set `OPENAI_API_KEY` in the environment or `.env` before model calls."
    if item == "env:FIRECRAWL":
        return "Set `FIRECRAWL_API_KEY` or `FIRECRAWL_API_URL` if MAScan web search should run."
    if item == "env:NEWS_API_KEY":
        return "Set `NEWS_API_KEY` if the political news tool should run."
    if item == "env:TWITTER_COOKIES":
        return "Set `TWITTER_AUTH_TOKEN` and `TWITTER_CT0` if the social X search tool should run."
    if item == "pricing_file" and "zero" in message:
        return "Replace zero placeholder rates in the pricing file with the cited pricing snapshot."
    if item == "pricing_file" and "source_url" in message:
        return "Add the source URL for the pricing snapshot to the pricing file."
    if item == "pricing_file" and "captured_at" in message:
        return "Add the date/time the pricing snapshot was captured."
    if item == "pricing_file":
        return "Create or fix the pricing file referenced by the manifest."
    if item == "gold_standard_file":
        return "Check the manifest `gold_standard_file` path and JSON schema."
    if item.startswith("source_pdf:"):
        return "Restore or correct the referenced PDF path in the gold-standard dataset."
    if item == "human_calibration":
        return "Add the `human_calibration` section to the experiment manifest."
    if item == "human.packet_file":
        return "Create or restore the human packet before importing returned ratings."
    if item == "human.rater_ids":
        return "Define the expected human rater IDs in the manifest."
    if item == "human.ratings_file":
        return "Provide returned rater CSV files or create the manifest ratings JSON file."
    if item == "ratings_csv":
        return "Fix the returned ratings CSV files and rerun post-human preflight."
    return "Fix the listed issue and rerun preflight."


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _check_modules(
    module_names: tuple[str, ...],
    issues: list[PreflightIssue],
    *,
    severity: Severity = "error",
) -> None:
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            _add_issue(
                issues,
                severity,
                f"dependency:{module_name}",
                f"Missing Python module: {module_name}",
            )


def _check_openai_key(base: Path, issues: list[PreflightIssue]) -> None:
    value = _env_value("OPENAI_API_KEY", base)
    if not value or value == "sk-your-key-here":
        _add_issue(
            issues,
            "error",
            "env:OPENAI_API_KEY",
            "OPENAI_API_KEY is required for response collection and judging.",
        )


def _check_mascan_tool_credentials(base: Path, issues: list[PreflightIssue]) -> None:
    if not _env_value("FIRECRAWL_API_KEY", base) and not _env_value("FIRECRAWL_API_URL", base):
        _add_issue(
            issues,
            "warning",
            "env:FIRECRAWL",
            "MAScan web_search needs FIRECRAWL_API_KEY or FIRECRAWL_API_URL.",
        )
    if not _env_value("NEWS_API_KEY", base):
        _add_issue(
            issues,
            "warning",
            "env:NEWS_API_KEY",
            "Political news_api tool may fail without NEWS_API_KEY.",
        )
    if not _env_value("TWITTER_AUTH_TOKEN", base) or not _env_value("TWITTER_CT0", base):
        _add_issue(
            issues,
            "warning",
            "env:TWITTER_COOKIES",
            "Social x_search may be unavailable without TWITTER_AUTH_TOKEN and TWITTER_CT0.",
        )


def _check_gold_standard(
    manifest: GoldExperimentManifest,
    base: Path,
    issues: list[PreflightIssue],
) -> None:
    path = _resolve(base, manifest.gold_standard_file)
    if not path.exists():
        _add_issue(
            issues,
            "error",
            "gold_standard_file",
            f"Missing gold standard file: {manifest.gold_standard_file}",
        )
        return
    try:
        dataset = GoldStandardDataset.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _add_issue(
            issues,
            "error",
            "gold_standard_file",
            f"Could not parse gold standard file: {exc}",
        )
        return
    if len(dataset.cases) != manifest.expected_case_count:
        _add_issue(
            issues,
            "error",
            "gold_standard_file",
            f"Expected {manifest.expected_case_count} cases, found {len(dataset.cases)}.",
        )
    for case in dataset.cases:
        if not _resolve(base, case.source_pdf).exists():
            _add_issue(
                issues,
                "error",
                f"source_pdf:{case.case_id}",
                f"Missing source PDF: {case.source_pdf}",
            )


def _check_pricing_table(
    manifest: GoldExperimentManifest,
    base: Path,
    issues: list[PreflightIssue],
) -> None:
    if not manifest.pricing_file:
        _add_issue(
            issues,
            "error",
            "pricing_file",
            "Manifest must define pricing_file before running the priced analysis.",
        )
        return
    path = _resolve(base, manifest.pricing_file)
    if not path.exists():
        _add_issue(
            issues,
            "error",
            "pricing_file",
            f"Missing pricing file: {manifest.pricing_file}",
        )
        return
    try:
        pricing = PricingTable.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _add_issue(
            issues,
            "error",
            "pricing_file",
            f"Could not parse pricing file: {exc}",
        )
        return
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


def _check_post_human_inputs(
    manifest: GoldExperimentManifest,
    base: Path,
    ratings_csv_files: list[str] | None,
    issues: list[PreflightIssue],
) -> None:
    human = manifest.human_calibration
    if human is None:
        _add_issue(
            issues,
            "error",
            "human_calibration",
            "Manifest is missing human_calibration.",
        )
        return
    packet = _load_human_packet(human.packet_file, base, issues)
    if ratings_csv_files:
        rows: list[dict[str, str]] = []
        missing_csv = False
        for csv_file in ratings_csv_files:
            csv_path = _resolve(base, csv_file)
            if not csv_path.exists():
                missing_csv = True
                _add_issue(
                    issues,
                    "error",
                    "ratings_csv",
                    f"Missing returned rater CSV: {csv_file}",
                )
                continue
            try:
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    rows.extend(dict(row) for row in reader)
            except Exception as exc:
                _add_issue(
                    issues,
                    "error",
                    "ratings_csv",
                    f"Could not read returned rater CSV {csv_file}: {exc}",
                )
        if not missing_csv and rows:
            try:
                ratings = human_ratings_from_csv_rows(rows)
            except Exception as exc:
                _add_issue(
                    issues,
                    "error",
                    "ratings_csv",
                    f"Could not parse returned ratings CSV rows: {exc}",
                )
            else:
                _validate_returned_ratings(
                    ratings,
                    packet,
                    human.rater_ids,
                    issues,
                    "ratings_csv",
                )
        elif not missing_csv:
            _add_issue(
                issues,
                "error",
                "ratings_csv",
                "Returned ratings CSV files contain no rating rows.",
            )
    elif human.ratings_file:
        ratings_path = _resolve(base, human.ratings_file)
        if not ratings_path.exists():
            _add_issue(
                issues,
                "error",
                "human.ratings_file",
                (
                    "No --ratings-csv files were provided and manifest ratings_file "
                    f"does not exist: {human.ratings_file}"
                ),
            )
            return
        try:
            ratings = HumanRatingsFile.model_validate_json(
                ratings_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            _add_issue(
                issues,
                "error",
                "human.ratings_file",
                f"Could not parse human ratings file {human.ratings_file}: {exc}",
            )
        else:
            _validate_returned_ratings(
                ratings,
                packet,
                human.rater_ids,
                issues,
                "human.ratings_file",
            )
    else:
        _add_issue(
            issues,
            "error",
            "human.ratings_file",
            "No --ratings-csv files were provided and manifest ratings_file is not defined.",
        )


def _load_human_packet(
    packet_file: str,
    base: Path,
    issues: list[PreflightIssue],
) -> HumanCalibrationPacket | None:
    packet_path = _resolve(base, packet_file)
    if not packet_path.exists():
        _add_issue(
            issues,
            "error",
            "human.packet_file",
            f"Missing human packet: {packet_file}",
        )
        return None
    try:
        return HumanCalibrationPacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        _add_issue(
            issues,
            "error",
            "human.packet_file",
            f"Could not parse human packet {packet_file}: {exc}",
        )
        return None


def _validate_returned_ratings(
    ratings: HumanRatingsFile,
    packet: HumanCalibrationPacket | None,
    rater_ids: list[str],
    issues: list[PreflightIssue],
    item: str,
) -> None:
    if packet is None:
        return
    if not rater_ids:
        _add_issue(
            issues,
            "error",
            "human.rater_ids",
            "Manifest must define expected rater_ids before validating returned ratings.",
        )
        return
    validation = validate_complete_human_ratings(
        ratings,
        packet,
        rater_ids=rater_ids,
    )
    if not validation.is_complete:
        _add_issue(
            issues,
            "error",
            item,
            "Returned human ratings are incomplete or inconsistent: "
            f"{validation.model_dump(mode='json')}",
        )


def _env_value(name: str, base: Path) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip().strip('"')
    env_path = base / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        if key.strip() == name:
            return raw_value.split("#", 1)[0].strip().strip('"')
    return None


def _resolve(base: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _add_issue(
    issues: list[PreflightIssue],
    severity: Severity,
    item: str,
    message: str,
) -> None:
    issues.append(PreflightIssue(severity=severity, item=item, message=message))
