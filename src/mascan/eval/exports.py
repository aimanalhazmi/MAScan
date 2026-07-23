"""Human-friendly exports for gold-standard prompts and freeze manifests."""

import csv
import hashlib
import io
import json
from collections.abc import Sequence

from mascan.eval.fingerprints import model_sha256
from mascan.eval.gold_judge import (
    GOLD_JUDGE_SYSTEM_PROMPT,
    build_gold_judge_user_prompt,
    gold_judge_output_schema,
    gold_judge_prompt_sha256,
    gold_judge_schema_sha256,
)
from mascan.eval.gold_standard import (
    EXPECTED_OUTPUT_FIELDS,
    PESTEL_HEADINGS,
    GoldStandardCase,
    GoldStandardCoverageReport,
    GoldStandardDataset,
    validate_gold_standard_coverage,
)

PROMPT_PACK_CSV_FIELDS = ["case_id", "case_title", "source_pdf", "prompt"]
GOLD_STANDARD_MANIFEST_CSV_FIELDS = [
    "case_id",
    "case_title",
    "source_pdf",
    "prompt_sha256",
    "expected_output_sha256",
    "gold_claims_sha256",
    "category_targets_sha256",
    "gold_claim_count",
    "category_target_count",
    "source_anchor_count",
    "expected_sections_complete",
    "category_targets_cover_all_buckets",
]


def render_prompt_pack_markdown(dataset: GoldStandardDataset) -> str:
    lines = [
        "# Gold-Standard PESTEL Prompt Pack",
        "",
        "Use the same prompt for every control group.",
    ]
    for case in dataset.cases:
        lines += [
            "",
            "---",
            "",
            f"## {case.case_id}: {case.case_title}",
            "",
            f"**Source PDF:** `{case.source_pdf}`",
            "",
            "```text",
            case.prompt,
            "```",
        ]
    return "\n".join(lines) + "\n"


def render_gold_standard_validation_report(
    dataset: GoldStandardDataset,
    coverage_report: GoldStandardCoverageReport | None = None,
) -> str:
    """Render prompt, expected answer, and reread justification for each case."""
    coverage_report = coverage_report or validate_gold_standard_coverage(dataset)
    all_expected_sections_present = all(
        all(getattr(case.expected_output, field) for field in EXPECTED_OUTPUT_FIELDS)
        for case in dataset.cases
    )
    all_target_buckets_present = all(
        {target.correct_category for target in case.category_targets} == set(PESTEL_HEADINGS)
        for case in dataset.cases
    )
    target_counts = [len(case.category_targets) for case in dataset.cases]
    claim_counts = [len(case.gold_claims) for case in dataset.cases]

    lines = [
        "# Gold-Standard Dataset Validation Report",
        "",
        "This report exposes the prompt, expected PESTEL answer, source anchors, "
        "and reread justification for every case in the frozen dataset.",
        "",
        "## Consistency Summary",
        "",
        f"- Cases: {len(dataset.cases)}",
        f"- PDF papers in inventory: {coverage_report.pdf_count}",
        f"- Dataset/PDF coverage exact: {_yes_no(coverage_report.is_valid)}",
        f"- Expected output sections complete: {_yes_no(all_expected_sections_present)}",
        f"- Categorization targets cover all PESTEL buckets per case: "
        f"{_yes_no(all_target_buckets_present)}",
        f"- Categorization targets per case: {min(target_counts)}-{max(target_counts)}",
        f"- Gold causal claims per case: {min(claim_counts)}-{max(claim_counts)}",
    ]

    if coverage_report.issues:
        lines += ["", "## Coverage Issues", ""]
        for issue in coverage_report.issues:
            case_prefix = f"{issue.case_id}: " if issue.case_id else ""
            lines.append(f"- {issue.code}: {case_prefix}{issue.detail}")

    for case in dataset.cases:
        lines += [
            "",
            "---",
            "",
            f"## {case.case_id}: {case.case_title}",
            "",
            f"**Source PDF:** `{case.source_pdf}`",
            "",
            f"**Case Subject:** {case.case_subject}",
            "",
            "### Prompt",
            "",
            "```text",
            case.prompt,
            "```",
            "",
            "### Expected Output",
            "",
        ]
        for field in EXPECTED_OUTPUT_FIELDS:
            label = field.replace("_", " ").title()
            lines += [f"**{label}**", ""]
            lines.extend(f"- {bullet}" for bullet in getattr(case.expected_output, field))
            lines.append("")

        lines += ["### Categorization Targets", ""]
        for target in case.category_targets:
            lines.append(f"- {target.factor} -> {target.correct_category}: {target.rationale}")

        lines += ["", "### Source Anchors", ""]
        lines.extend(f"- {anchor}" for anchor in case.validation_notes.source_anchors)

        lines += [
            "",
            "### Reread Justification",
            "",
            case.validation_notes.reread_justification,
        ]

        if case.avoid_claims:
            lines += ["", "### Avoid Claims", ""]
            lines.extend(f"- {claim}" for claim in case.avoid_claims)

    return "\n".join(lines).rstrip() + "\n"


def render_gold_judge_rubric_markdown(
    *,
    sample_case: GoldStandardCase | None = None,
    sample_response_text: str = "[MODEL RESPONSE TEXT GOES HERE]",
) -> str:
    """Render the strict judge prompt and structured-output contract."""
    lines = [
        "# Gold-Standard PESTEL Judge Rubric",
        "",
        "This is the exact system prompt and structured-output contract used by "
        "the LLM-as-a-judge pipeline for the 25-case PESTEL gold standard.",
        "",
        "## Fingerprints",
        "",
        f"- `judge_prompt_sha256`: `{gold_judge_prompt_sha256()}`",
        f"- `judge_schema_sha256`: `{gold_judge_schema_sha256()}`",
        "",
        "## System Prompt",
        "",
        "```text",
        GOLD_JUDGE_SYSTEM_PROMPT.rstrip(),
        "```",
        "",
        "## Structured Output Fields",
        "",
    ]

    schema = gold_judge_output_schema()
    properties = schema.get("properties", {})
    for field_name in [
        "response_claim_scores",
        "category_judgments",
        "missing_gold_claims",
        "unsupported_or_wrong_claims",
        "summary",
    ]:
        field_schema = properties.get(field_name, {})
        description = field_schema.get("description") or field_schema.get("title") or ""
        lines.append(f"- `{field_name}`: {description}".rstrip())

    lines += [
        "",
        "## Full Structured Output Schema",
        "",
        "```json",
        json.dumps(schema, indent=2, ensure_ascii=False),
        "```",
    ]

    if sample_case is not None:
        lines += [
            "",
            "## Sample Case-Specific User Prompt",
            "",
            f"Case: `{sample_case.case_id}`",
            "",
            "```text",
            build_gold_judge_user_prompt(sample_case, sample_response_text).rstrip(),
            "```",
        ]

    return "\n".join(lines).rstrip() + "\n"


def prompt_pack_csv_rows(dataset: GoldStandardDataset) -> list[dict[str, str]]:
    return [
        {
            "case_id": case.case_id,
            "case_title": case.case_title,
            "source_pdf": case.source_pdf,
            "prompt": case.prompt,
        }
        for case in dataset.cases
    ]


def gold_standard_manifest_payload(
    dataset: GoldStandardDataset,
    coverage_report: GoldStandardCoverageReport | None = None,
) -> dict[str, object]:
    """Return a compact freeze manifest for the gold-standard dataset."""
    coverage_report = coverage_report or validate_gold_standard_coverage(dataset)
    rows = _gold_standard_manifest_rows(dataset)
    return {
        "schema_version": dataset.schema_version,
        "dataset_sha256": model_sha256(dataset),
        "case_count": len(dataset.cases),
        "pdf_count": coverage_report.pdf_count,
        "coverage_valid": coverage_report.is_valid,
        "coverage_issue_count": len(coverage_report.issues),
        "cases": rows,
    }


def gold_standard_manifest_csv_rows(
    dataset: GoldStandardDataset,
) -> list[dict[str, str]]:
    """Return per-case freeze manifest rows suitable for CSV export."""
    return [
        {key: _csv_value(row[key]) for key in GOLD_STANDARD_MANIFEST_CSV_FIELDS}
        for row in _gold_standard_manifest_rows(dataset)
    ]


def render_gold_standard_manifest_markdown(
    dataset: GoldStandardDataset,
    coverage_report: GoldStandardCoverageReport | None = None,
) -> str:
    """Render a compact case-level reproducibility manifest."""
    payload = gold_standard_manifest_payload(dataset, coverage_report)
    lines = [
        "# Gold-Standard Dataset Freeze Manifest",
        "",
        f"- Dataset SHA-256: `{payload['dataset_sha256']}`",
        f"- Cases: {payload['case_count']}",
        f"- PDF papers in inventory: {payload['pdf_count']}",
        f"- Dataset/PDF coverage valid: {_yes_no(bool(payload['coverage_valid']))}",
        f"- Coverage issues: {payload['coverage_issue_count']}",
        "",
        "| Case | Prompt SHA-256 | Expected Output SHA-256 | Claims | Targets | Anchors | Sections | Buckets |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    cases = payload["cases"]
    assert isinstance(cases, list)
    for row in cases:
        assert isinstance(row, dict)
        lines.append(
            "| "
            f"{row['case_id']} | "
            f"`{row['prompt_sha256']}` | "
            f"`{row['expected_output_sha256']}` | "
            f"{row['gold_claim_count']} | "
            f"{row['category_target_count']} | "
            f"{row['source_anchor_count']} | "
            f"{_yes_no(bool(row['expected_sections_complete']))} | "
            f"{_yes_no(bool(row['category_targets_cover_all_buckets']))} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _gold_standard_manifest_rows(
    dataset: GoldStandardDataset,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected_buckets = set(PESTEL_HEADINGS)
    for case in dataset.cases:
        expected_output = case.expected_output.model_dump(mode="json")
        gold_claims = [claim.model_dump(mode="json") for claim in case.gold_claims]
        category_targets = [target.model_dump(mode="json") for target in case.category_targets]
        rows.append(
            {
                "case_id": case.case_id,
                "case_title": case.case_title,
                "source_pdf": case.source_pdf,
                "prompt_sha256": _text_sha256(case.prompt),
                "expected_output_sha256": _json_sha256(expected_output),
                "gold_claims_sha256": _json_sha256(gold_claims),
                "category_targets_sha256": _json_sha256(category_targets),
                "gold_claim_count": len(gold_claims),
                "category_target_count": len(category_targets),
                "source_anchor_count": len(case.validation_notes.source_anchors),
                "expected_sections_complete": all(
                    bool(getattr(case.expected_output, field)) for field in EXPECTED_OUTPUT_FIELDS
                ),
                "category_targets_cover_all_buckets": {
                    target.correct_category for target in case.category_targets
                }
                == expected_buckets,
            }
        )
    return rows


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _csv_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def csv_text(rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()
