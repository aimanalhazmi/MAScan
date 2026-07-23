"""Loader and typed contracts for the hand-built PESTEL gold-standard dataset."""

from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_GOLD_STANDARD_PATH = Path("eval_papers/gold_standard_cases.json")

PESTEL_HEADINGS = (
    "Political",
    "Economic",
    "Social",
    "Technological",
    "Environmental",
    "Legal",
)
EXPECTED_OUTPUT_FIELDS = (
    "political",
    "economic",
    "social",
    "technological",
    "environmental",
    "legal",
    "strategic_implications",
)


class ExpectedPestelOutput(BaseModel):
    political: list[str] = Field(default_factory=list)
    economic: list[str] = Field(default_factory=list)
    social: list[str] = Field(default_factory=list)
    technological: list[str] = Field(default_factory=list)
    environmental: list[str] = Field(default_factory=list)
    legal: list[str] = Field(default_factory=list)
    strategic_implications: list[str] = Field(default_factory=list)


class GoldClaim(BaseModel):
    category: str
    claim: str


class CategoryTarget(BaseModel):
    factor: str
    correct_category: str
    rationale: str


class ValidationNotes(BaseModel):
    source_anchors: list[str] = Field(default_factory=list)
    reread_justification: str


class GoldStandardCase(BaseModel):
    case_id: str
    source_pdf: str
    case_title: str
    case_subject: str
    prompt: str
    expected_output: ExpectedPestelOutput
    gold_claims: list[GoldClaim] = Field(default_factory=list)
    category_targets: list[CategoryTarget] = Field(default_factory=list)
    avoid_claims: list[str] = Field(default_factory=list)
    validation_notes: ValidationNotes


class GoldStandardDataset(BaseModel):
    schema_version: str
    created_at: str
    purpose: str
    generation_instruction_template: str
    rubric_support: dict[str, object]
    cases: list[GoldStandardCase]

    def by_id(self, case_id: str) -> GoldStandardCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"Unknown gold-standard case_id: {case_id}")


class GoldStandardValidationIssue(BaseModel):
    code: str
    detail: str
    case_id: str | None = None


class GoldStandardCoverageReport(BaseModel):
    case_count: int
    pdf_count: int
    expected_case_count: int
    source_pdfs: list[str]
    inventory_pdfs: list[str]
    issues: list[GoldStandardValidationIssue] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


def load_gold_standard(
    path: str | Path = DEFAULT_GOLD_STANDARD_PATH,
) -> GoldStandardDataset:
    """Load the frozen 25-case PESTEL gold standard from disk."""
    dataset_path = Path(path)
    return GoldStandardDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))


def prompt_pack(dataset: GoldStandardDataset) -> list[dict[str, str]]:
    """Return prompt-only records for model/control-group generation."""
    return [
        {
            "case_id": case.case_id,
            "case_title": case.case_title,
            "source_pdf": case.source_pdf,
            "prompt": case.prompt,
        }
        for case in dataset.cases
    ]


def validate_gold_standard_coverage(
    dataset: GoldStandardDataset,
    papers_dir: str | Path = "eval_papers",
    *,
    expected_case_count: int = 25,
) -> GoldStandardCoverageReport:
    """Validate that the frozen dataset covers the paper inventory exactly."""
    papers_path = Path(papers_dir)
    inventory_pdfs = sorted(path.as_posix() for path in papers_path.glob("*.pdf"))
    inventory_names = {Path(path).name for path in inventory_pdfs}
    source_pdfs = [Path(case.source_pdf).as_posix() for case in dataset.cases]
    source_names = [Path(path).name for path in source_pdfs]
    issues: list[GoldStandardValidationIssue] = []

    if len(dataset.cases) != expected_case_count:
        issues.append(
            GoldStandardValidationIssue(
                code="case_count_mismatch",
                detail=(f"Expected {expected_case_count} cases, found {len(dataset.cases)}."),
            )
        )

    if len(inventory_pdfs) != expected_case_count:
        issues.append(
            GoldStandardValidationIssue(
                code="pdf_count_mismatch",
                detail=(
                    f"Expected {expected_case_count} PDF papers, found "
                    f"{len(inventory_pdfs)} in {papers_path.as_posix()}."
                ),
            )
        )

    _append_duplicate_issues(issues, [case.case_id for case in dataset.cases], "case_id")
    _append_duplicate_issues(issues, source_names, "source_pdf")

    missing_sources = sorted(set(source_names) - inventory_names)
    for filename in missing_sources:
        issues.append(
            GoldStandardValidationIssue(
                code="source_pdf_missing_from_inventory",
                detail=f"Dataset references {filename}, but it is not in the PDF inventory.",
            )
        )

    unrepresented_papers = sorted(inventory_names - set(source_names))
    for filename in unrepresented_papers:
        issues.append(
            GoldStandardValidationIssue(
                code="paper_missing_from_dataset",
                detail=f"PDF inventory contains {filename}, but no dataset case references it.",
            )
        )

    case_ids = [case.case_id for case in dataset.cases]
    if case_ids != sorted(case_ids):
        issues.append(
            GoldStandardValidationIssue(
                code="case_order_not_stable",
                detail="Cases should remain sorted by case_id for deterministic exports.",
            )
        )

    expected_categories = set(PESTEL_HEADINGS)
    for case in dataset.cases:
        _validate_case(case, dataset, expected_categories, issues)

    return GoldStandardCoverageReport(
        case_count=len(dataset.cases),
        pdf_count=len(inventory_pdfs),
        expected_case_count=expected_case_count,
        source_pdfs=source_pdfs,
        inventory_pdfs=inventory_pdfs,
        issues=issues,
    )


def _validate_case(
    case: GoldStandardCase,
    dataset: GoldStandardDataset,
    expected_categories: set[str],
    issues: list[GoldStandardValidationIssue],
) -> None:
    if Path(case.source_pdf).stem != case.case_id:
        issues.append(
            GoldStandardValidationIssue(
                code="case_id_source_pdf_mismatch",
                case_id=case.case_id,
                detail=(
                    f"case_id {case.case_id} does not match source PDF stem "
                    f"{Path(case.source_pdf).stem}."
                ),
            )
        )

    expected_prompt = dataset.generation_instruction_template.format(case_subject=case.case_subject)
    if case.prompt != expected_prompt:
        issues.append(
            GoldStandardValidationIssue(
                code="prompt_template_mismatch",
                case_id=case.case_id,
                detail="Prompt does not match the frozen generation template.",
            )
        )

    for field in EXPECTED_OUTPUT_FIELDS:
        if not getattr(case.expected_output, field):
            issues.append(
                GoldStandardValidationIssue(
                    code="expected_output_section_empty",
                    case_id=case.case_id,
                    detail=f"Expected output section {field} is empty.",
                )
            )

    claim_categories = {claim.category for claim in case.gold_claims}
    invalid_claim_categories = sorted(claim_categories - expected_categories)
    for category in invalid_claim_categories:
        issues.append(
            GoldStandardValidationIssue(
                code="invalid_gold_claim_category",
                case_id=case.case_id,
                detail=f"Gold claim category {category} is not a PESTEL heading.",
            )
        )

    if len(case.gold_claims) < 5:
        issues.append(
            GoldStandardValidationIssue(
                code="too_few_gold_claims",
                case_id=case.case_id,
                detail="Each case needs at least five causal claims for depth judging.",
            )
        )

    target_categories = {target.correct_category for target in case.category_targets}
    if target_categories != expected_categories:
        issues.append(
            GoldStandardValidationIssue(
                code="category_target_bucket_gap",
                case_id=case.case_id,
                detail=(
                    "Category targets must cover exactly the six PESTEL headings; "
                    f"found {sorted(target_categories)}."
                ),
            )
        )

    for target in case.category_targets:
        if target.correct_category not in expected_categories:
            issues.append(
                GoldStandardValidationIssue(
                    code="invalid_category_target_bucket",
                    case_id=case.case_id,
                    detail=(
                        f"Target {target.factor} uses invalid bucket {target.correct_category}."
                    ),
                )
            )

    if not case.validation_notes.source_anchors:
        issues.append(
            GoldStandardValidationIssue(
                code="missing_source_anchors",
                case_id=case.case_id,
                detail="Validation notes need source anchors from the reread pass.",
            )
        )

    if not case.validation_notes.reread_justification.strip():
        issues.append(
            GoldStandardValidationIssue(
                code="missing_reread_justification",
                case_id=case.case_id,
                detail="Validation notes need a reread justification.",
            )
        )


def _append_duplicate_issues(
    issues: list[GoldStandardValidationIssue], values: list[str], label: str
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    for value in sorted(duplicates):
        issues.append(
            GoldStandardValidationIssue(
                code=f"duplicate_{label}",
                detail=f"Duplicate {label}: {value}.",
            )
        )
