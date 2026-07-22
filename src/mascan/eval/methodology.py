"""Methodology and status appendix for the gold-standard PESTEL experiment."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from mascan.eval.readiness import (
    GoldExperimentManifest,
    ReadinessIssue,
    ReadinessReport,
)


class MethodologyChecklistItem(BaseModel):
    step: int
    title: str
    status: str
    evidence: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _StepDefinition:
    step: int
    title: str
    requirement: str


_STEPS = (
    _StepDefinition(
        1,
        "Control Groups",
        "Run MAScan, zero-shot same-model, and frontier-model systems on the same frozen prompts.",
    ),
    _StepDefinition(
        2,
        "Gold-Standard Dataset",
        "Use 25 case-study prompts, expected PESTEL outputs, gold claims, category targets, and reread notes.",
    ),
    _StepDefinition(
        3,
        "Metrics",
        "Score analytical depth on a 1-3 rubric and categorization accuracy as strict target-bucket accuracy; report grounding accuracy separately as a secondary diagnostic.",
    ),
    _StepDefinition(
        4,
        "LLM-as-a-Judge",
        "Enumerate causal claims, mechanically average claim-level depth scores, and judge all category targets; grounding is flagged in an isolated second pass over the same enumerated claims.",
    ),
    _StepDefinition(
        5,
        "Trace And Cost Analysis",
        "Report token usage, cost, latency, quality per 1k tokens, and quality per USD.",
    ),
    _StepDefinition(
        6,
        "Statistical Significance",
        "Compare MAScan to both baselines using paired t-test or Wilcoxon signed-rank based on the normality protocol.",
    ),
)


def build_methodology_checklist(
    manifest: GoldExperimentManifest,
    readiness: ReadinessReport | None = None,
) -> list[MethodologyChecklistItem]:
    issue_map = _issues_by_step(readiness.issues if readiness else [])
    items: list[MethodologyChecklistItem] = []
    for step in _STEPS:
        issues = issue_map.get(step.step, [])
        status = _status_for_issues(issues)
        items.append(
            MethodologyChecklistItem(
                step=step.step,
                title=step.title,
                status=status,
                evidence=_evidence_for_step(manifest, step.step),
                issues=[f"{issue.item}: {issue.message}" for issue in issues],
            )
        )
    return items


def render_methodology_appendix(
    manifest: GoldExperimentManifest,
    *,
    readiness: ReadinessReport | None = None,
) -> str:
    checklist = build_methodology_checklist(manifest, readiness)
    lines = [
        "# Gold-Standard PESTEL Evaluation Methodology Appendix",
        "",
        "## Protocol Summary",
        "",
        "- Dataset: 25 case studies converted into frozen prompts, expected PESTEL outputs, gold causal claims, categorization targets, avoid-claims, source anchors, and reread justifications.",
        "- Control groups: MAScan, zero-shot same-model, and frontier-model outputs are generated from the same prompt text; prompt hashes are checked before readiness passes.",
        "- Analytical Depth: each distinct causal claim in a response is scored 1, 2, or 3, then averaged mechanically.",
        "- Categorization Accuracy: each gold target factor is counted correct only when present and primarily mapped to its expected PESTEL bucket.",
        "- Combined Quality: the unweighted mean of normalized analytical depth and categorization accuracy. This is the primary metric and is unchanged across all runs.",
        "- Grounding Accuracy: a secondary reported diagnostic, not a term in Combined Quality. It is the share of the enumerated response claims that a separate grounding pass did not flag as ungrounded, where ungrounded means contradicting the case, restating an avoid-claim, contradicting another claim in the same answer, or being false or anachronistic for the case period. Rate-based rather than a raw count so it does not scale with verbosity.",
        "- Grounding is judged in its own call with its own prompt hash (`grounding_prompt_sha256`), never appended to the depth/categorization prompt. An earlier revision appended it to the shared prompt and silently re-scored analytical depth on all five smoke cases for responses whose text had not changed, so the two rubrics are kept isolated to preserve comparability.",
        "- The grounding rubric defaults to grounded. The gold-standard case is a short summary rather than a corpus, so it cannot support arbitrary factual particulars; an earlier revision that treated 'not supported by the case evidence' as ungrounded flagged 26 of 27 correct claims on one case and systematically penalised specificity while rewarding vague hedged prose. Absence from the gold claims is therefore never treated as evidence of a grounding problem.",
        "- Cost and trace analysis: priced judged records are flattened into per-case rows with quality, token, latency, and cost diagnostics.",
        "- Significance testing: paired system scores are compared case-by-case; Shapiro-Wilk normality is used when available with the manifest alpha threshold, otherwise Wilcoxon signed-rank is the conservative default.",
        "",
        "## Readiness Checklist",
        "",
        "| Step | Requirement | Status | Evidence | Issues |",
        "|---:|---|---|---|---|",
    ]
    for item in checklist:
        requirement = _STEPS[item.step - 1].requirement
        lines.append(
            "| "
            f"{item.step} | "
            f"{requirement} | "
            f"{item.status} | "
            f"{_join_table_cell(item.evidence)} | "
            f"{_join_table_cell(item.issues)} |"
        )

    if readiness is not None:
        if readiness.fingerprints:
            lines += [
                "",
                "## Reproducibility Fingerprints",
                "",
                "| Artifact | Method | SHA-256 |",
                "|---|---|---|",
            ]
            for fingerprint in readiness.fingerprints:
                lines.append(
                    "| "
                    f"{fingerprint.artifact} | "
                    f"{fingerprint.method} | "
                    f"`{fingerprint.sha256}` |"
                )

        lines += [
            "",
            "## Readiness Result",
            "",
            f"- Ready: {str(readiness.is_ready).lower()}",
            f"- Errors: {readiness.errors}",
            f"- Warnings: {readiness.warnings}",
        ]

    return "\n".join(lines) + "\n"


def write_methodology_appendix(
    path: str | Path,
    manifest: GoldExperimentManifest,
    *,
    readiness: ReadinessReport | None = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_methodology_appendix(manifest, readiness=readiness),
        encoding="utf-8",
    )


def _issues_by_step(issues: list[ReadinessIssue]) -> dict[int, list[ReadinessIssue]]:
    grouped: dict[int, list[ReadinessIssue]] = {step.step: [] for step in _STEPS}
    for issue in issues:
        grouped[_step_for_issue(issue.item)].append(issue)
    return grouped


def _step_for_issue(item: str) -> int:
    if item.startswith(("response_file:", "merged_responses_file")):
        return 1
    if item.startswith(("gold_standard_file", "source_pdf:", "gold_targets:")):
        return 2
    if item.startswith(("judged_file", "priced_judged_file")):
        return 4
    if item.startswith(("pricing_file", "system_summary_file", "case_trace_file")):
        return 5
    if item.startswith(("comparison:",)):
        return 6
    if item.startswith("final_report_file"):
        return 6
    return 3


def _status_for_issues(issues: list[ReadinessIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "incomplete"
    if any(issue.severity == "warning" for issue in issues):
        return "warning"
    return "complete"


def _evidence_for_step(manifest: GoldExperimentManifest, step: int) -> list[str]:
    if step == 1:
        return [
            *(system.response_file for system in manifest.systems),
            *( [manifest.merged_responses_file] if manifest.merged_responses_file else [] ),
        ]
    if step == 2:
        return [manifest.gold_standard_file]
    if step == 3:
        return ["src/mascan/eval/gold_judge.py", "src/mascan/eval/gold_experiment.py"]
    if step == 4:
        return [path for path in [manifest.judged_file, manifest.priced_judged_file] if path]
    if step == 5:
        return [
            path
            for path in [
                manifest.pricing_file,
                manifest.system_summary_file,
                manifest.case_trace_file,
            ]
            if path
        ]
    if step == 6:
        return [
            *(comparison.file for comparison in manifest.comparisons),
            *( [manifest.final_report_file] if manifest.final_report_file else [] ),
        ]
    return []


def _join_table_cell(values: list[str]) -> str:
    if not values:
        return "-"
    return "<br>".join(value.replace("|", "\\|") for value in values)
