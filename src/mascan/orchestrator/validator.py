import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from mascan.contracts.reports import AgentReport
from mascan.core.logging import get_logger
from mascan.orchestrator.state import GraphState

logger = get_logger("orchestrator.validator")

HTML_SOURCE_REF_PATTERN = re.compile(r'href=["\']#source-(\d+)["\']')
MARKDOWN_SOURCE_REF_PATTERN = re.compile(r"\[(\d+)\]\(([^)]+)\)")

ValidationStatus = Literal["passed", "warnings", "failed to validate"]
ValidationCategory = Literal[
    "unsupported_claim",
    "source_mismatch",
    "agent_disagreement",
    "citation_gap",
    "uncertain_or_stale_data",
]
ValidationSeverity = Literal["low", "medium", "high"]

VALIDATOR_SYSTEM_PROMPT = """\
You are the validation reviewer for a PESTEL multi-agent market-analysis system.
Your job is to compare the final synthesized report against the supplied agent
reports and sources only.

Do not use outside knowledge, web browsing, or new assumptions. Flag only issues
that can be identified from the provided evidence. Focus on:
- claims in the final report that are not supported by agent reports,
- claims that contradict agent reports or source metadata,
- disagreements between agents that the final report hides,
- missing citations for important factual claims,
- stale, sparse, or uncertain data presented too confidently.

When flagging a claim that has a numbered citation, copy the exact final-report
passage into the claim field including its citation marker, for example:
[1](https://source-url)
When the issue is a citation gap, copy the uncited passage and explain that no
source number is attached.

Return concise validation findings. If no obvious issues are present, return an
empty issues list and a brief overall note.
"""


class ValidationIssue(BaseModel):
    """A single validation finding."""

    category: ValidationCategory
    severity: ValidationSeverity
    claim: str = Field(
        description=(
            "The exact final-report claim or passage being reviewed, including any "
            "numbered citation marker copied from the report."
        )
    )
    explanation: str = Field(description="Why this claim may be unsupported or disputed.")
    relevant_agents: list[str] = Field(
        default_factory=list,
        description="Agent reports or sources relevant to this issue.",
    )


class ValidationResult(BaseModel):
    """Structured output from the final report validator."""

    issues: list[ValidationIssue] = Field(default_factory=list)
    overall_note: str


def validator_node(state: GraphState) -> dict[str, Any]:
    """Validate the synthesized final report and append a Fact Check section."""
    try:
        result = run_validation(state)
        validation_markdown = render_validation_markdown(result)
        validation_payload = validation_payload_from_result(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Final report validation failed")
        validation_markdown = render_validation_failure(exc)
        validation_payload = {
            "status": "failed to validate",
            "issues": [],
            "overall_note": "Validation could not be completed.",
            "error": f"{type(exc).__name__}: {exc}",
        }

    final_markdown = append_fact_check(state.final_markdown, validation_markdown)
    return {
        "final_markdown": final_markdown,
        "validation_status": validation_payload["status"],
        "validation_issues": validation_payload["issues"],
        "validation_markdown": validation_markdown,
        "validation_payload": validation_payload,
    }


def run_validation(state: GraphState) -> ValidationResult:
    from langchain_core.messages import HumanMessage, SystemMessage

    structured_llm = get_validation_model()
    result: ValidationResult = structured_llm.invoke(
        [
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=build_validation_prompt(state)),
        ]
    )
    return result


def get_validation_model() -> Any:
    from mascan.core.llm import get_chat_model
    from mascan.core.settings import get_settings

    settings = get_settings()
    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.0,
        max_tokens=1200,
    )
    return llm.with_structured_output(ValidationResult)


def build_validation_prompt(state: GraphState) -> str:
    parts = [
        f"User question:\n{state.user_input}\n",
        "Final report to validate:\n",
        state.final_markdown or state.final_summary or "(empty final report)",
        "\nAgent evidence:\n",
    ]
    if state.reports:
        for name, report in state.reports.items():
            parts.append(format_agent_report(name, report))
    else:
        parts.append("(no successful agent reports)\n")

    if state.failures:
        parts.append("Agent failures:\n")
        for name, error in state.failures.items():
            parts.append(f"- {name}: {error}\n")

    return "\n".join(parts)


def format_agent_report(name: str, report: AgentReport) -> str:
    sources = "\n".join(
        f"  - {source.name}: {source.url}" if source.url else f"  - {source.name}"
        for source in report.sources
    )
    return (
        f"### Agent: {name}\n"
        f"Confidence: {report.confidence:.2f}\n"
        f"Findings:\n{report.findings}\n"
        f"Sources:\n{sources or '  (no sources)'}\n"
    )


def render_validation_markdown(result: ValidationResult) -> str:
    status = validation_status_from_result(result)
    lines = ["## Fact Check", "", f"**Status:** {status}", "", result.overall_note]
    if not result.issues:
        lines.extend(["", "No obvious factual issues were detected from the provided sources."])
        return "\n".join(lines)

    lines.extend(["", "### Issues"])
    for index, issue in enumerate(result.issues, start=1):
        agents = ", ".join(issue.relevant_agents) or "not specified"
        citations = render_issue_citations(issue)
        lines.extend(
            [
                f"{index}. **{issue.severity} / {issue.category}**",
                f"   - Claim: {issue.claim}",
                f"   - Citation(s): {citations}",
                f"   - Why it matters: {issue.explanation}",
                f"   - Relevant evidence: {agents}",
            ]
        )
    return "\n".join(lines)


def render_validation_failure(exc: Exception) -> str:
    return (
        "## Fact Check\n\n"
        "**Status:** failed to validate\n\n"
        f"Validation could not be completed: {type(exc).__name__}: {exc}"
    )


def append_fact_check(markdown: str, validation_markdown: str) -> str:
    base = markdown.rstrip() if markdown else "# Final Report"
    return f"{base}\n\n{validation_markdown}\n"


def validation_payload_from_result(result: ValidationResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    payload["status"] = validation_status_from_result(result)
    return payload


def validation_status_from_result(result: ValidationResult) -> ValidationStatus:
    return "warnings" if result.issues else "passed"


def render_issue_citations(issue: ValidationIssue) -> str:
    links = extract_source_links(f"{issue.claim}\n{issue.explanation}")
    if links:
        return " ".join(f"[{number}]({url})" for number, url in links)
    numbers = extract_html_source_numbers(f"{issue.claim}\n{issue.explanation}")
    if numbers:
        return " ".join(f"[{number}]" for number in numbers)
    if issue.category == "citation_gap":
        return "missing"
    return "not specified"


def extract_source_links(text: str) -> list[tuple[int, str]]:
    links: list[tuple[int, str]] = []
    seen: set[int] = set()
    for match in MARKDOWN_SOURCE_REF_PATTERN.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            links.append((number, match.group(2)))
    return links


def extract_html_source_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for match in HTML_SOURCE_REF_PATTERN.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers
