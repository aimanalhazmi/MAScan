"""Component-owned execution measurement helpers."""

import time
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core import callbacks

from mascan.contracts.metrics import AgentCallMetrics, ComponentMetrics, TokenUsage


def _summarize_usage(
    usage_by_model: Mapping[str, Mapping[str, Any]],
) -> TokenUsage:
    input_tokens = sum(
        int(usage.get("input_tokens", 0)) for usage in usage_by_model.values()
    )
    output_tokens = sum(
        int(usage.get("output_tokens", 0)) for usage in usage_by_model.values()
    )
    total_tokens = sum(
        int(usage.get("total_tokens", 0)) for usage in usage_by_model.values()
    )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
    )


def aggregate_component_token_usage(
    component_metrics: Mapping[str, ComponentMetrics],
) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(
            metric.token_usage.input_tokens for metric in component_metrics.values()
        ),
        output_tokens=sum(
            metric.token_usage.output_tokens for metric in component_metrics.values()
        ),
        total_tokens=sum(
            metric.token_usage.total_tokens for metric in component_metrics.values()
        ),
    )


def aggregate_agent_token_usage(
    agent_metrics: Mapping[str, AgentCallMetrics],
) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(
            metric.token_usage.input_tokens for metric in agent_metrics.values()
        ),
        output_tokens=sum(
            metric.token_usage.output_tokens for metric in agent_metrics.values()
        ),
        total_tokens=sum(
            metric.token_usage.total_tokens for metric in agent_metrics.values()
        ),
    )


def measure_agent_call[T](
    name: str,
    operation: Callable[[], T],
) -> tuple[T, dict[str, AgentCallMetrics]]:
    with callbacks.get_usage_metadata_callback(
        name=f"mascan_agent_{name}"
    ) as usage_callback:
        result = operation()
    return result, {
        name: AgentCallMetrics(
            run_count=1,
            token_usage=_summarize_usage(usage_callback.usage_metadata),
        )
    }


def measure_component(
    name: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    with callbacks.get_usage_metadata_callback(
        name=f"mascan_component_{name}"
    ) as usage_callback:
        update = operation()
    return {
        **update,
        "component_metrics": {
            name: ComponentMetrics(
                run_count=1,
                duration_seconds=round(time.perf_counter() - started_at, 6),
                token_usage=_summarize_usage(usage_callback.usage_metadata),
            )
        },
    }
