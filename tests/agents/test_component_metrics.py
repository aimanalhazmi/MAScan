from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

import mascan.agents.base as base_module
from mascan.agents.economics.agent import EconomicsAgent
from mascan.agents.environmental.agent import EnvironmentalAgent
from mascan.agents.legal.agent import LegalAgent
from mascan.agents.political.agent import PoliticalAgent
from mascan.agents.social.agent import SocialAgent, SocialEvidencePlan
from mascan.agents.technological.agent import TechnologicalAgent
from mascan.contracts.tools import ToolResult


@pytest.mark.parametrize(
    ("agent_type", "react_result"),
    [
        (EconomicsAgent, ({}, "findings", [])),
        (EnvironmentalAgent, ({}, "findings", [])),
        (LegalAgent, ({}, "findings", [])),
        (PoliticalAgent, ({}, "findings", [])),
        (SocialAgent, ({}, "findings", [], [])),
        (TechnologicalAgent, ({}, "findings", [])),
    ],
)
def test_private_agent_subgraphs_return_owned_metrics(
    mocker: Any,
    agent_type: type[Any],
    react_result: tuple[Any, ...],
) -> None:
    agent = agent_type()
    tasks = ["original task"]
    context = {"runtime": {"current_date": "2026-07-15"}}
    call_order: list[str] = []

    def gather_deterministic(*args: Any, **kwargs: Any) -> dict[str, Any]:
        call_order.append("gather")
        return {}

    gather = mocker.patch.object(
        agent,
        "gather_deterministic",
        side_effect=gather_deterministic,
    )
    original_build_initial_state = agent.build_initial_state
    mocker.patch.object(
        agent,
        "build_initial_state",
        side_effect=lambda *args, **kwargs: (
            call_order.append("private_graph"),
            original_build_initial_state(*args, **kwargs),
        )[1],
    )
    mocker.patch.object(agent, "run_react_agent", return_value=react_result)

    @contextmanager
    def fake_usage_callback(*args: Any, **kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(usage_metadata={})

    mocker.patch(
        "mascan.core.metrics.callbacks.get_usage_metadata_callback",
        fake_usage_callback,
    )
    if agent_type is EconomicsAgent:
        mocker.patch.object(
            base_module,
            "time",
            SimpleNamespace(perf_counter=mocker.Mock(side_effect=[10.0, 15.0])),
            create=True,
        )

    report = agent.run(tasks, context=context)

    metric = report.component_metrics[agent.name]
    assert metric.run_count == 1
    assert metric.agents["analyst"].run_count == 1
    assert metric.token_usage == metric.agents["analyst"].token_usage
    gather.assert_called_once()
    gather_call = gather.call_args
    assert gather_call.args == (tasks,)
    assert gather_call.kwargs["context"] is context
    assert gather_call.kwargs["agent_metrics"] == {}
    assert call_order == ["gather", "private_graph"]
    if agent_type is EconomicsAgent:
        assert metric.duration_seconds == 5.0
    else:
        assert metric.duration_seconds >= 0
    assert "execution" not in report.metadata


def test_social_agent_metrics_include_only_planner_and_analyst_llm_calls(
    mocker: Any,
) -> None:
    agent = SocialAgent()
    plan = SocialEvidencePlan(country_codes=["DEU"], web_queries=[])
    mocker.patch.object(agent, "plan_evidence", return_value=plan)
    world_bank = mocker.patch.object(
        agent.tools["world_bank_social_indicators"],
        "run",
        return_value=ToolResult(
            success=True,
            data=[],
            source="world_bank:social_indicators",
        ),
    )
    mocker.patch.object(
        agent,
        "run_react_agent",
        return_value=({}, "findings", [], []),
    )
    callback_names: list[str] = []

    @contextmanager
    def fake_usage_callback(*args: Any, **kwargs: Any) -> Iterator[Any]:
        name = kwargs["name"]
        callback_names.append(name)
        total_tokens = {
            "mascan_agent_evidence_planner": 20,
            "mascan_agent_analyst": 80,
        }.get(name, 0)
        yield SimpleNamespace(
            usage_metadata={
                "gpt-test": {
                    "input_tokens": total_tokens,
                    "output_tokens": 0,
                    "total_tokens": total_tokens,
                }
            }
        )

    mocker.patch(
        "mascan.core.metrics.callbacks.get_usage_metadata_callback",
        fake_usage_callback,
    )

    report = agent.run(["German workforce trends"])

    social = report.component_metrics["social"]
    assert social.agents["evidence_planner"].token_usage.total_tokens == 20
    assert social.agents["analyst"].token_usage.total_tokens == 80
    assert social.token_usage.total_tokens == 100
    assert callback_names == [
        "mascan_agent_evidence_planner",
        "mascan_agent_analyst",
    ]
    world_bank.assert_called_once_with(country_codes=["DEU"])
