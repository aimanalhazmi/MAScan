"""Strict LLM-as-a-judge prompt for the 25-case PESTEL gold standard."""

import hashlib
import json
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator

from mascan.eval.gold_standard import GoldStandardCase

GOLD_JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluation judge for PESTEL market-analysis outputs.
Judge only the submitted system answer against the provided gold-standard case.
Do not reward polish, length, or generic business knowledge unless it maps to the
case evidence.

Analytical Depth metric:
1. Enumerate every distinct causal relationship the submitted answer makes.
   A causal relationship has the shape "external factor X affects business
   outcome Y" or "factor X implies strategic action/risk/opportunity Y".
2. Score each enumerated response claim mechanically:
   - 1 Surface: the answer merely mentions an external factor without connecting
     it to the business, operating model, financials, compliance, demand,
     reputation, or strategy.
   - 2 Analytical: the answer names the factor and explicitly explains a direct
     operational, financial, demand, compliance, or reputational impact.
   - 3 Strategic: the answer names the factor, traces the operational impact,
     and connects it to a strategic risk, opportunity, recommendation, or shift.
3. If the answer contains no causal relationship claims, return an empty
   response_claim_scores list; the evaluator will assign the minimum score.

Categorization Accuracy metric:
For every category target, determine whether the answer discusses the same
factor or a clear equivalent. If present, record the PESTEL heading where the
answer placed it. Mark correct only when it is present and primarily mapped to
the expected category. Missing factors are incorrect for strict accuracy.

Use exactly one category_judgment for each provided category target, preserving
the target factor text and order.
"""

GROUNDING_JUDGE_SYSTEM_PROMPT = """\
You are a strict factual-grounding checker for PESTEL market-analysis outputs.
You receive a gold-standard case, and a list of claims extracted from a submitted
answer. Decide which claims are ungrounded.

Default to GROUNDED. The gold-standard case is a short summary, not a corpus: it
cannot possibly restate every true fact about the subject, period, or geography.
A specific, plausible, period-correct fact is GROUNDED even when the case summary
never mentions it. Absence from the gold claims is NEVER evidence of a problem.

Mark a claim UNGROUNDED only when one of these is demonstrably true:
   - contradiction: it conflicts with the case context, the expected output, or a
     gold claim.
   - avoid_claim: it restates or endorses an entry in avoid_claims.
   - inconsistency: it contradicts another claim in the same submitted answer.
   - false_or_anachronistic: from your own world knowledge it is simply wrong, or
     it refers to a regulation, organisation, technology, event, or figure that did
     not exist or did not apply in the case's time frame and geography.

Never mark a claim ungrounded for being specific, quantitative, or named. Naming a
real regulation, statistic, price, or organisation that fits the period is exactly
what a good analysis does, and specificity must never be penalised. Do not mark a
claim merely because you cannot personally verify a figure: only mark it when you
have positive reason to believe it is wrong.

Quote each ungrounded claim verbatim, give the reason kind, and explain briefly.
Return an empty list when every claim is grounded.
"""


class ResponseClaimScore(BaseModel):
    response_claim: str
    category: str | None = None
    linked_gold_claims: list[str] = Field(default_factory=list)
    score: int = Field(..., ge=1, le=3)
    reasoning: str


class CategoryTargetJudgment(BaseModel):
    factor: str
    expected_category: str
    observed_category: str | None = None
    present: bool
    correct: bool
    reasoning: str


class _LLMGoldJudgeOutput(BaseModel):
    response_claim_scores: list[ResponseClaimScore] = Field(default_factory=list)
    category_judgments: list[CategoryTargetJudgment] = Field(default_factory=list)
    missing_gold_claims: list[str] = Field(default_factory=list)
    unsupported_or_wrong_claims: list[str] = Field(default_factory=list)
    summary: str


class UngroundedClaim(BaseModel):
    claim: str
    kind: Literal[
        "contradiction",
        "avoid_claim",
        "inconsistency",
        "false_or_anachronistic",
    ]
    reasoning: str


class _LLMGroundingOutput(BaseModel):
    ungrounded_claims: list[UngroundedClaim] = Field(default_factory=list)


class GroundingJudgeResult(BaseModel):
    """Output of the separate grounding pass.

    Judged in its own call so that adding or tuning the grounding rubric cannot
    change analytical-depth or categorization scores, which are produced by an
    untouched prompt and stay comparable across runs.
    """

    case_id: str
    ungrounded_claims: list[UngroundedClaim] = Field(default_factory=list)
    claims_reviewed: int = Field(default=0, ge=0)
    ungrounded_claim_count: int = Field(default=0, ge=0)
    grounding_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    judge_model: str
    grounding_prompt_sha256: str

    @model_validator(mode="after")
    def populate_counts(self) -> "GroundingJudgeResult":
        self.ungrounded_claim_count = len(self.ungrounded_claims)
        return self


class GoldJudgeResult(_LLMGoldJudgeOutput):
    case_id: str
    analytical_depth_score: float = Field(..., ge=1.0, le=3.0)
    categorization_accuracy: float = Field(..., ge=0.0, le=1.0)
    categorization_accuracy_present_only: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    grounding: GroundingJudgeResult | None = Field(
        default=None,
        description=(
            "Result of the separate grounding pass. None when grounding was not "
            "assessed for this record."
        ),
    )
    judge_model: str
    judge_prompt_sha256: str
    judge_schema_sha256: str
    response_claim_count: int = Field(default=0, ge=0)
    unsupported_claim_count: int = Field(default=0, ge=0)
    category_targets_evaluated: int = Field(default=0, ge=0)
    category_targets_present: int = Field(default=0, ge=0)
    category_targets_missing: int = Field(default=0, ge=0)
    category_targets_correct: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def populate_metric_counts(self) -> "GoldJudgeResult":
        self.response_claim_count = len(self.response_claim_scores)
        self.unsupported_claim_count = len(self.unsupported_or_wrong_claims)
        self.category_targets_evaluated = len(self.category_judgments)
        self.category_targets_present = sum(
            1 for judgment in self.category_judgments if judgment.present
        )
        self.category_targets_missing = (
            self.category_targets_evaluated - self.category_targets_present
        )
        self.category_targets_correct = sum(
            1 for judgment in self.category_judgments if judgment.correct
        )
        return self


def get_chat_model(
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> Any:
    """Lazy wrapper so pure metric tests do not require LangChain on import."""
    from mascan.core.llm import get_chat_model as _get_chat_model

    return _get_chat_model(model=model, temperature=temperature, max_tokens=max_tokens)


def _default_judge_model() -> str:
    from mascan.core.settings import get_settings

    return get_settings().eval_judge_model


def _judge_messages(system_prompt: str, user_prompt: str) -> list[object]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ModuleNotFoundError:
        # Unit tests with mocked models do not need LangChain message objects.
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def compute_analytical_depth(
    response_claim_scores: list[ResponseClaimScore],
) -> float:
    """Mechanical average over enumerated response-claim scores."""
    if not response_claim_scores:
        return 1.0
    return round(
        sum(claim.score for claim in response_claim_scores)
        / len(response_claim_scores),
        4,
    )


def compute_grounding_accuracy(claims_reviewed: int, ungrounded_count: int) -> float:
    """Share of reviewed claims the grounding pass did not flag as ungrounded.

    Rate-based rather than a raw count: an unbounded count scales with verbosity,
    so a terse answer would look artificially well grounded. An answer with no
    claims earns no grounding credit — grounding is earned by making supported
    claims, not by staying silent.
    """
    if claims_reviewed <= 0:
        return 0.0
    return round(max(0.0, 1.0 - ungrounded_count / claims_reviewed), 4)


def compute_categorization_accuracy(
    judgments: list[CategoryTargetJudgment],
    *,
    present_only: bool = False,
) -> float | None:
    """Share of category targets mapped to the expected PESTEL bucket."""
    denominator = [j for j in judgments if j.present] if present_only else judgments
    if not denominator:
        return None if present_only else 0.0
    return round(sum(1 for j in denominator if j.correct) / len(denominator), 4)


def validate_category_judgment_alignment(
    case: GoldStandardCase,
    judgments: list[CategoryTargetJudgment],
) -> None:
    """Require one judge categorization judgment per gold target, in order."""
    expected = [
        (target.factor, target.correct_category)
        for target in case.category_targets
    ]
    observed = [
        (judgment.factor, judgment.expected_category)
        for judgment in judgments
    ]
    if observed != expected:
        raise ValueError(
            "Judge category_judgments must preserve the gold category_targets "
            f"exactly for {case.case_id}. "
            f"Expected {len(expected)} ordered targets, received {len(observed)}."
        )


def gold_judge_output_schema() -> dict[str, Any]:
    """Return the structured-output schema required from the LLM judge."""
    return _LLMGoldJudgeOutput.model_json_schema()


def gold_judge_prompt_sha256() -> str:
    """Hash the exact judge system prompt used for rubric evaluation."""
    return hashlib.sha256(GOLD_JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def grounding_judge_prompt_sha256() -> str:
    """Hash the grounding rubric so grounding runs are reproducible on their own."""
    return hashlib.sha256(GROUNDING_JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def gold_judge_schema_sha256() -> str:
    """Hash the judge structured-output schema as canonical JSON."""
    canonical = json.dumps(
        gold_judge_output_schema(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _expected_output_payload(case: GoldStandardCase) -> dict[str, list[str]]:
    return case.expected_output.model_dump(mode="json")


def build_gold_judge_user_prompt(case: GoldStandardCase, response_text: str) -> str:
    """Build a single-case judge prompt with gold targets and the model response."""
    gold_claims = [claim.model_dump(mode="json") for claim in case.gold_claims]
    category_targets = [
        target.model_dump(mode="json") for target in case.category_targets
    ]
    payload = {
        "case_id": case.case_id,
        "case_title": case.case_title,
        "case_subject": case.case_subject,
        "generation_prompt": case.prompt,
        "expected_output": _expected_output_payload(case),
        "gold_claims": gold_claims,
        "category_targets": category_targets,
        "avoid_claims": case.avoid_claims,
    }
    return (
        "## Gold-standard case\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "## Submitted answer to evaluate\n"
        f"{response_text}\n"
    )


def build_grounding_judge_user_prompt(
    case: GoldStandardCase,
    response_claims: list[str],
) -> str:
    """Build the grounding prompt from the case and the already-enumerated claims.

    Reuses the main judge's claim enumeration so both passes share one denominator
    and the two calls cannot disagree about what the answer actually claimed.
    """
    payload = {
        "case_id": case.case_id,
        "case_title": case.case_title,
        "case_subject": case.case_subject,
        "generation_prompt": case.prompt,
        "expected_output": _expected_output_payload(case),
        "gold_claims": [claim.model_dump(mode="json") for claim in case.gold_claims],
        "avoid_claims": case.avoid_claims,
    }
    claim_lines = "\n".join(
        f"{index}. {claim}" for index, claim in enumerate(response_claims, start=1)
    )
    return (
        "## Gold-standard case\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "## Claims extracted from the submitted answer\n"
        f"{claim_lines}\n"
    )


def judge_grounding(
    case: GoldStandardCase,
    response_claims: list[str],
    model: str | None = None,
) -> GroundingJudgeResult:
    """Run the standalone grounding pass over already-enumerated response claims."""
    model = model or _default_judge_model()
    if not response_claims:
        return GroundingJudgeResult(
            case_id=case.case_id,
            claims_reviewed=0,
            grounding_accuracy=0.0,
            judge_model=model,
            grounding_prompt_sha256=grounding_judge_prompt_sha256(),
        )

    llm = get_chat_model(model=model, temperature=0, max_tokens=2000)
    structured = llm.with_structured_output(_LLMGroundingOutput)
    out = cast(
        _LLMGroundingOutput,
        structured.invoke(
            _judge_messages(
                GROUNDING_JUDGE_SYSTEM_PROMPT,
                build_grounding_judge_user_prompt(case, response_claims),
            )
        ),
    )
    return GroundingJudgeResult(
        case_id=case.case_id,
        ungrounded_claims=out.ungrounded_claims,
        claims_reviewed=len(response_claims),
        grounding_accuracy=compute_grounding_accuracy(
            len(response_claims), len(out.ungrounded_claims)
        ),
        judge_model=model,
        grounding_prompt_sha256=grounding_judge_prompt_sha256(),
    )


def parse_gold_judge_output(payload: dict[str, Any]) -> "_LLMGoldJudgeOutput":
    """Validate a raw judge payload produced outside this process.

    Used when a second judge (for inter-rater reliability) is run through an
    external tool rather than the API, so its output is held to exactly the same
    schema as the in-process judge.
    """
    return _LLMGoldJudgeOutput.model_validate(payload)


def build_gold_judge_result(
    case: GoldStandardCase,
    out: "_LLMGoldJudgeOutput",
    *,
    model: str,
    grounding: GroundingJudgeResult | None = None,
) -> GoldJudgeResult:
    """Assemble the scored result from a raw judge output, whatever produced it."""
    validate_category_judgment_alignment(case, out.category_judgments)
    return GoldJudgeResult(
        case_id=case.case_id,
        grounding=grounding,
        response_claim_scores=out.response_claim_scores,
        category_judgments=out.category_judgments,
        missing_gold_claims=out.missing_gold_claims,
        unsupported_or_wrong_claims=out.unsupported_or_wrong_claims,
        summary=out.summary,
        analytical_depth_score=compute_analytical_depth(out.response_claim_scores),
        categorization_accuracy=compute_categorization_accuracy(out.category_judgments)
        or 0.0,
        categorization_accuracy_present_only=compute_categorization_accuracy(
            out.category_judgments, present_only=True
        ),
        judge_model=model,
        judge_prompt_sha256=gold_judge_prompt_sha256(),
        judge_schema_sha256=gold_judge_schema_sha256(),
    )


def judge_gold_response(
    case: GoldStandardCase,
    response_text: str,
    model: str | None = None,
    *,
    include_grounding: bool = False,
) -> GoldJudgeResult:
    """Judge one generated PESTEL response against one gold-standard case.

    Grounding is an opt-in second call: it is a reported secondary diagnostic, not
    part of combined quality, and it must not touch the depth/categorization prompt.
    """
    model = model or _default_judge_model()
    llm = get_chat_model(model=model, temperature=0, max_tokens=4000)
    structured = llm.with_structured_output(_LLMGoldJudgeOutput)
    out = cast(
        _LLMGoldJudgeOutput,
        structured.invoke(
            _judge_messages(
                GOLD_JUDGE_SYSTEM_PROMPT,
                build_gold_judge_user_prompt(case, response_text),
            )
        ),
    )
    validate_category_judgment_alignment(case, out.category_judgments)
    grounding = (
        judge_grounding(
            case,
            [claim.response_claim for claim in out.response_claim_scores],
            model=model,
        )
        if include_grounding
        else None
    )
    return build_gold_judge_result(case, out, model=model, grounding=grounding)
