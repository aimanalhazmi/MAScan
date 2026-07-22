"""Structured output for citation-pair validation."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ValidationStatus = Literal["passed", "warnings", "failed_to_validate"]
ValidationCategory = Literal["inaccessible_source", "relevant_content", "fact_check"]
RelevantContentStatus = Literal[
    "relevant",
    "partially_relevant",
    "unrelated",
    "uncertain",
]
RelevantIssueSubtype = Literal["partially_relevant", "unrelated", "uncertain"]
FactCheckStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "uncertain",
]
FactCheckIssueSubtype = Literal[
    "partially_supported",
    "unsupported",
    "contradicted",
    "uncertain",
]
CheckStatus = Literal["passed", "issue", "failed"]
StoppedAfter = Literal["link_works", "relevant_content", "fact_check"]


class ValidationCitation(BaseModel):
    number: int
    url: str


class ValidationIssue(BaseModel):
    """One user-facing problem found for a citation-claim pair."""

    category: ValidationCategory
    subtype: RelevantIssueSubtype | FactCheckIssueSubtype | None = None
    claim: str
    passage: str
    citation: ValidationCitation
    explanation: str

    @model_validator(mode="after")
    def validate_category_subtype(self) -> "ValidationIssue":
        relevant_subtypes = {"partially_relevant", "unrelated", "uncertain"}
        fact_subtypes = {
            "partially_supported",
            "unsupported",
            "contradicted",
            "uncertain",
        }
        if self.category == "inaccessible_source" and self.subtype is not None:
            raise ValueError("inaccessible_source must not have a subtype")
        if self.category == "relevant_content" and self.subtype not in relevant_subtypes:
            raise ValueError("relevant_content requires a relevance subtype")
        if self.category == "fact_check" and self.subtype not in fact_subtypes:
            raise ValueError("fact_check requires a fact-check subtype")
        return self


class CitationCheck(BaseModel):
    """Auditable result for one citation-claim pair."""

    status: CheckStatus
    claim: str
    passage: str
    citation: ValidationCitation
    link_works: bool
    relevant_content: RelevantContentStatus | None = None
    fact_check: FactCheckStatus | None = None
    stopped_after: StoppedAfter
    explanation: str
    error: str | None = None

    @model_validator(mode="after")
    def validate_staged_result(self) -> "CitationCheck":
        if self.status == "failed":
            if not self.error:
                raise ValueError("failed checks require an operational error")
            return self
        if not self.link_works:
            if self.status != "issue":
                raise ValueError("inaccessible sources must be reported as issues")
            if self.stopped_after != "link_works":
                raise ValueError("inaccessible checks must stop after link_works")
            if self.relevant_content is not None or self.fact_check is not None:
                raise ValueError("inaccessible checks cannot contain judge results")
            return self
        if self.relevant_content != "relevant":
            if self.status != "issue" or self.stopped_after != "relevant_content":
                raise ValueError("relevance problems must stop as issues")
            if self.fact_check is not None:
                raise ValueError("Fact Check must not run after a relevance problem")
            return self
        if self.fact_check is None or self.stopped_after != "fact_check":
            raise ValueError("relevant sources require a Fact Check result")
        expected_status = "passed" if self.fact_check == "supported" else "issue"
        if self.status != expected_status:
            raise ValueError("check status does not match the Fact Check result")
        return self


class ValidationSummary(BaseModel):
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    issues: int = Field(ge=0)
    failed: int = Field(ge=0)


class ValidationReport(BaseModel):
    """Complete result emitted by the Validator graph node."""

    status: ValidationStatus
    summary: ValidationSummary
    issues: list[ValidationIssue] = Field(default_factory=list)
    checks: list[CitationCheck] = Field(default_factory=list)
    markdown: str = ""
    error: str | None = None
