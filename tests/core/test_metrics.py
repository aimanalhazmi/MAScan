from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from mascan.contracts.metrics import (
    AgentCallMetrics,
    ComponentMetrics,
    TokenUsage,
    merge_agent_metrics,
    merge_component_metrics,
)
from mascan.core.metrics import (
    aggregate_agent_token_usage,
    aggregate_component_token_usage,
    measure_agent_call,
    measure_component,
)


def test_agent_metrics_accumulate_repeated_calls() -> None:
    merged = merge_agent_metrics(
        {
            "analyst": AgentCallMetrics(
                run_count=1,
                token_usage=TokenUsage(
                    input_tokens=8,
                    output_tokens=2,
                    total_tokens=10,
                ),
            )
        },
        {
            "analyst": AgentCallMetrics(
                run_count=1,
                token_usage=TokenUsage(
                    input_tokens=12,
                    output_tokens=3,
                    total_tokens=15,
                ),
            )
        },
    )

    assert merged["analyst"] == AgentCallMetrics(
        run_count=2,
        token_usage=TokenUsage(
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
        ),
    )


def test_merge_agent_metrics_copies_left_only_records() -> None:
    original = AgentCallMetrics(
        run_count=1,
        token_usage=TokenUsage(
            input_tokens=8,
            output_tokens=2,
            total_tokens=10,
        ),
    )

    merged = merge_agent_metrics({"analyst": original}, {})

    assert merged["analyst"] == original
    assert merged["analyst"] is not original
    assert merged["analyst"].token_usage is not original.token_usage


def test_component_metrics_accumulate_repeated_planner_runs() -> None:
    merged = merge_component_metrics(
        {
            "planner": ComponentMetrics(
                run_count=1,
                duration_seconds=1.5,
                token_usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                ),
                agents={
                    "analyst": AgentCallMetrics(
                        run_count=1,
                        token_usage=TokenUsage(
                            input_tokens=8,
                            output_tokens=2,
                            total_tokens=10,
                        ),
                    )
                },
            )
        },
        {
            "planner": ComponentMetrics(
                run_count=1,
                duration_seconds=2.0,
                token_usage=TokenUsage(
                    input_tokens=7,
                    output_tokens=3,
                    total_tokens=10,
                ),
                agents={
                    "analyst": AgentCallMetrics(
                        run_count=1,
                        token_usage=TokenUsage(
                            input_tokens=12,
                            output_tokens=3,
                            total_tokens=15,
                        ),
                    )
                },
            )
        },
    )

    assert merged["planner"] == ComponentMetrics(
        run_count=2,
        duration_seconds=3.5,
        token_usage=TokenUsage(
            input_tokens=17,
            output_tokens=5,
            total_tokens=22,
        ),
        agents={
            "analyst": AgentCallMetrics(
                run_count=2,
                token_usage=TokenUsage(
                    input_tokens=20,
                    output_tokens=5,
                    total_tokens=25,
                ),
            )
        },
    )


def test_merge_component_metrics_copies_left_only_records() -> None:
    original = ComponentMetrics(
        run_count=1,
        duration_seconds=1.5,
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
        ),
        agents={
            "analyst": AgentCallMetrics(
                run_count=1,
                token_usage=TokenUsage(
                    input_tokens=8,
                    output_tokens=2,
                    total_tokens=10,
                ),
            )
        },
    )

    merged = merge_component_metrics({"planner": original}, {})

    assert merged["planner"] == original
    assert merged["planner"] is not original
    assert merged["planner"].token_usage is not original.token_usage
    assert merged["planner"].agents is not original.agents
    assert merged["planner"].agents["analyst"] is not original.agents["analyst"]
    assert (
        merged["planner"].agents["analyst"].token_usage
        is not original.agents["analyst"].token_usage
    )


def test_aggregate_agent_tokens_sums_each_agent_once() -> None:
    usage = aggregate_agent_token_usage(
        {
            "analyst": AgentCallMetrics(
                run_count=2,
                token_usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                ),
            ),
            "reviewer": AgentCallMetrics(
                run_count=1,
                token_usage=TokenUsage(
                    input_tokens=20,
                    output_tokens=5,
                    total_tokens=25,
                ),
            ),
        }
    )

    assert usage == TokenUsage(
        input_tokens=30,
        output_tokens=7,
        total_tokens=37,
    )


def test_aggregate_component_tokens_sums_each_component_once() -> None:
    usage = aggregate_component_token_usage(
        {
            "planner": ComponentMetrics(
                run_count=2,
                token_usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                ),
            ),
            "economics": ComponentMetrics(
                run_count=1,
                token_usage=TokenUsage(
                    input_tokens=20,
                    output_tokens=5,
                    total_tokens=25,
                ),
            ),
        }
    )

    assert usage == TokenUsage(
        input_tokens=30,
        output_tokens=7,
        total_tokens=37,
    )


def test_measure_component_preserves_update_and_returns_typed_metrics(mocker: Any) -> None:
    @contextmanager
    def fake_usage_callback(*args: Any, **kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(
            usage_metadata={
                "gpt-test": {
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "total_tokens": 50,
                }
            }
        )

    mocker.patch(
        "mascan.core.metrics.callbacks.get_usage_metadata_callback",
        fake_usage_callback,
    )
    mocker.patch("mascan.core.metrics.time.perf_counter", side_effect=[3.0, 4.25])

    update = measure_component("planner", lambda: {"plan": {}})

    assert update["plan"] == {}
    assert update["component_metrics"] == {
        "planner": ComponentMetrics(
            run_count=1,
            duration_seconds=1.25,
            token_usage=TokenUsage(
                input_tokens=40,
                output_tokens=10,
                total_tokens=50,
            ),
        )
    }


def test_measure_agent_call_collects_tokens_without_duration(mocker: Any) -> None:
    @contextmanager
    def fake_usage_callback(*args: Any, **kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(
            usage_metadata={
                "gpt-test": {
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "total_tokens": 50,
                }
            }
        )

    mocker.patch(
        "mascan.core.metrics.callbacks.get_usage_metadata_callback",
        fake_usage_callback,
    )

    result, metrics = measure_agent_call("analyst", lambda: "answer")

    assert result == "answer"
    assert metrics["analyst"].run_count == 1
    assert metrics["analyst"].token_usage.total_tokens == 50
    assert not hasattr(metrics["analyst"], "duration_seconds")
