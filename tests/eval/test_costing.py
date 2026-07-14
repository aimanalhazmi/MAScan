import pytest

from mascan.eval.costing import (
    ModelPricing,
    PricingTable,
    apply_pricing_to_judged,
    apply_pricing_to_responses,
    estimate_response_cost_usd,
    pricing_template_for_models,
    validate_pricing_table,
)
from mascan.eval.gold_experiment import JudgedModelResponse, ModelResponseRecord, TokenUsage
from mascan.eval.readiness import load_experiment_manifest


def test_estimate_response_cost_usd_uses_prompt_and_completion_rates():
    cost = estimate_response_cost_usd(
        TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000),
        ModelPricing(
            model="m",
            prompt_usd_per_1m_tokens=2.0,
            completion_usd_per_1m_tokens=6.0,
        ),
    )

    assert cost == 5.0


def test_apply_pricing_to_responses_sets_cost():
    records = [
        ModelResponseRecord(
            case_id="c",
            system_id="s",
            model="m",
            token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=2000),
        )
    ]
    pricing = PricingTable(
        prices=[
            ModelPricing(
                model="m",
                prompt_usd_per_1m_tokens=1.0,
                completion_usd_per_1m_tokens=2.0,
            )
        ]
    )

    priced = apply_pricing_to_responses(records, pricing)

    assert priced[0].token_usage.cost_usd == 0.005


def test_apply_pricing_to_responses_can_require_known_models():
    with pytest.raises(ValueError):
        apply_pricing_to_responses(
            [ModelResponseRecord(case_id="c", system_id="s", model="missing")],
            PricingTable(prices=[]),
        )


def test_apply_pricing_to_judged_updates_nested_response():
    judged = [
        JudgedModelResponse(
            response=ModelResponseRecord(
                case_id="c",
                system_id="s",
                model="m",
                token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=1000),
            )
        )
    ]
    pricing = PricingTable(
        prices=[
            ModelPricing(
                model="m",
                prompt_usd_per_1m_tokens=1.0,
                completion_usd_per_1m_tokens=1.0,
            )
        ]
    )

    priced = apply_pricing_to_judged(judged, pricing)

    assert priced[0].response.token_usage.cost_usd == 0.002


def test_pricing_template_for_models_deduplicates_and_uses_zero_placeholders():
    template = pricing_template_for_models(
        ["m2", "m1", "m1"],
        source_url="https://example.test/pricing",
        captured_at="2026-07-12",
    )

    assert template.currency == "USD"
    assert template.source_url == "https://example.test/pricing"
    assert template.captured_at == "2026-07-12"
    assert [price.model for price in template.prices] == ["m1", "m2"]
    assert all(price.prompt_usd_per_1m_tokens == 0.0 for price in template.prices)


def test_validate_pricing_table_reports_coverage_metadata_and_zero_rates():
    pricing = PricingTable(
        prices=[
            ModelPricing(
                model="m1",
                prompt_usd_per_1m_tokens=0.0,
                completion_usd_per_1m_tokens=0.0,
            ),
            ModelPricing(
                model="m1",
                prompt_usd_per_1m_tokens=1.0,
                completion_usd_per_1m_tokens=1.0,
            ),
        ]
    )

    report = validate_pricing_table(pricing, expected_models=["m1", "m2"])

    assert report.has_errors
    assert report.has_warnings
    assert report.missing_models == ["m2"]
    assert report.duplicate_models == ["m1"]
    assert report.zero_rate_models == ["m1"]
    assert report.missing_metadata == ["source_url", "captured_at"]


def test_repository_pricing_matches_example_manifest():
    manifest = load_experiment_manifest("eval_papers/gold_experiment_manifest.example.json")
    pricing = pricing_template_for_models(
        [system.model for system in manifest.systems],
        source_url="https://openai.com/api/pricing/",
        captured_at="2026-07-12",
    )
    pricing = pricing.model_copy(
        update={
            "prices": [
                price.model_copy(
                    update={
                        "prompt_usd_per_1m_tokens": 0.15,
                        "completion_usd_per_1m_tokens": 0.6,
                    }
                )
                for price in pricing.prices
            ]
        }
    )

    report = validate_pricing_table(
        pricing,
        expected_models=[system.model for system in manifest.systems],
    )

    assert report.has_errors is False
    assert report.has_warnings is False
