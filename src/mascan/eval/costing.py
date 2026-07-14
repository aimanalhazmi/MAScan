"""Apply explicit model-pricing tables to saved evaluation responses."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from mascan.eval.gold_experiment import JudgedModelResponse, ModelResponseRecord, TokenUsage


class ModelPricing(BaseModel):
    model: str
    prompt_usd_per_1m_tokens: float = Field(..., ge=0.0)
    completion_usd_per_1m_tokens: float = Field(..., ge=0.0)
    notes: str | None = None


class PricingTable(BaseModel):
    currency: str = "USD"
    source_url: str | None = None
    captured_at: str | None = None
    notes: str | None = None
    prices: list[ModelPricing] = Field(default_factory=list)

    def by_model(self) -> dict[str, ModelPricing]:
        return {price.model: price for price in self.prices}


class PricingValidationReport(BaseModel):
    expected_models: list[str]
    priced_models: list[str]
    missing_models: list[str] = Field(default_factory=list)
    zero_rate_models: list[str] = Field(default_factory=list)
    duplicate_models: list[str] = Field(default_factory=list)
    missing_metadata: list[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.missing_models or self.duplicate_models)

    @property
    def has_warnings(self) -> bool:
        return bool(self.zero_rate_models or self.missing_metadata)


def pricing_template_for_models(
    models: Sequence[str],
    *,
    source_url: str | None = None,
    captured_at: str | None = None,
    notes: str | None = None,
) -> PricingTable:
    """Create a schema-valid pricing table that must be manually filled."""
    return PricingTable(
        source_url=source_url,
        captured_at=captured_at,
        notes=notes
        or (
            "Fill prompt_usd_per_1m_tokens and completion_usd_per_1m_tokens "
            "with the pricing snapshot used for this experiment."
        ),
        prices=[
            ModelPricing(
                model=model,
                prompt_usd_per_1m_tokens=0.0,
                completion_usd_per_1m_tokens=0.0,
                notes="Replace zero placeholder rates before final cost analysis.",
            )
            for model in sorted(set(models))
        ],
    )


def validate_pricing_table(
    pricing_table: PricingTable,
    *,
    expected_models: Sequence[str],
    require_metadata: bool = True,
) -> PricingValidationReport:
    """Validate model coverage and citation metadata for cost analysis."""
    expected = sorted(set(expected_models))
    model_counts: dict[str, int] = {}
    for price in pricing_table.prices:
        model_counts[price.model] = model_counts.get(price.model, 0) + 1
    priced_models = sorted(model_counts)
    missing_metadata: list[str] = []
    if require_metadata:
        if not pricing_table.source_url:
            missing_metadata.append("source_url")
        if not pricing_table.captured_at:
            missing_metadata.append("captured_at")
    return PricingValidationReport(
        expected_models=expected,
        priced_models=priced_models,
        missing_models=sorted(set(expected) - set(priced_models)),
        zero_rate_models=sorted(
            price.model
            for price in pricing_table.prices
            if price.model in expected
            and price.prompt_usd_per_1m_tokens == 0.0
            and price.completion_usd_per_1m_tokens == 0.0
        ),
        duplicate_models=sorted(
            model for model, count in model_counts.items() if count > 1
        ),
        missing_metadata=missing_metadata,
    )


def estimate_response_cost_usd(
    usage: TokenUsage,
    pricing: ModelPricing,
) -> float | None:
    """Estimate USD cost from prompt/completion tokens and a model price."""
    if usage.prompt_tokens is None or usage.completion_tokens is None:
        return None
    prompt_cost = usage.prompt_tokens * pricing.prompt_usd_per_1m_tokens / 1_000_000
    completion_cost = (
        usage.completion_tokens * pricing.completion_usd_per_1m_tokens / 1_000_000
    )
    return round(prompt_cost + completion_cost, 8)


def apply_pricing_to_responses(
    records: Sequence[ModelResponseRecord],
    pricing_table: PricingTable,
    *,
    require_price: bool = True,
) -> list[ModelResponseRecord]:
    """Return response records with token_usage.cost_usd filled when possible."""
    by_model = pricing_table.by_model()
    priced: list[ModelResponseRecord] = []
    for record in records:
        pricing = by_model.get(record.model)
        if pricing is None:
            if require_price:
                raise ValueError(f"No pricing found for model: {record.model}")
            priced.append(record)
            continue
        cost = estimate_response_cost_usd(record.token_usage, pricing)
        priced.append(
            record.model_copy(
                update={
                    "token_usage": record.token_usage.model_copy(
                        update={"cost_usd": cost}
                    )
                }
            )
        )
    return priced


def apply_pricing_to_judged(
    records: Sequence[JudgedModelResponse],
    pricing_table: PricingTable,
    *,
    require_price: bool = True,
) -> list[JudgedModelResponse]:
    """Return judged records with response token_usage.cost_usd filled."""
    priced_responses = apply_pricing_to_responses(
        [record.response for record in records],
        pricing_table,
        require_price=require_price,
    )
    return [
        record.model_copy(update={"response": priced_response})
        for record, priced_response in zip(records, priced_responses, strict=True)
    ]
