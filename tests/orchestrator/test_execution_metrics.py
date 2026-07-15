from typing import Any

from langgraph.graph import END, START, StateGraph

from mascan.contracts.metrics import AgentCallMetrics, ComponentMetrics, TokenUsage
from mascan.orchestrator.graph import agents_passthrough, run
from mascan.orchestrator.planner import planner_node
from mascan.orchestrator.state import GraphState
from mascan.orchestrator.synthesizer import synthesizer_node
from mascan.orchestrator.validator import ValidationResult, validator_node


def test_parent_components_return_owned_metrics(mocker: Any) -> None:
    mocker.patch("mascan.orchestrator.planner.agent_registry.all_names", return_value=[])
    mocker.patch(
        "mascan.orchestrator.validator.run_validation",
        return_value=ValidationResult(issues=[], overall_note="Evidence is consistent."),
    )

    planner_update = planner_node(GraphState(user_input="question"))
    synthesizer_update = synthesizer_node(GraphState(user_input="question"))
    validator_update = validator_node(GraphState(user_input="question"))

    assert planner_update["component_metrics"]["planner"].run_count == 1
    assert synthesizer_update["component_metrics"]["synthesizer"].run_count == 1
    assert validator_update["component_metrics"]["validator"].run_count == 1


def test_agents_passthrough_does_not_remerge_component_metrics() -> None:
    planner_metric = ComponentMetrics(
        run_count=1,
        duration_seconds=1.5,
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
        ),
    )
    graph = StateGraph(GraphState)
    graph.add_node("agents", agents_passthrough)
    graph.add_edge(START, "agents")
    graph.add_edge("agents", END)

    result = graph.compile().invoke(
        GraphState(
            user_input="question",
            component_metrics={"planner": planner_metric},
        )
    )

    assert result["component_metrics"]["planner"] == planner_metric


def test_run_sums_component_tokens_and_measures_wall_time_independently(
    mocker: Any,
) -> None:
    state = {
        "user_input": "question",
        "final_summary": "answer",
        "final_markdown": "answer",
        "plan": {},
        "reports": {},
        "failures": {},
        "component_metrics": {
            "planner": ComponentMetrics(
                run_count=2,
                duration_seconds=4.0,
                token_usage=TokenUsage(
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                ),
            ),
            "economics": ComponentMetrics(
                run_count=1,
                duration_seconds=8.0,
                token_usage=TokenUsage(
                    input_tokens=200,
                    output_tokens=50,
                    total_tokens=250,
                ),
                agents={
                    "analyst": AgentCallMetrics(
                        run_count=1,
                        token_usage=TokenUsage(
                            input_tokens=200,
                            output_tokens=50,
                            total_tokens=250,
                        ),
                    )
                },
            ),
        },
        "validation_payload": {"status": "passed"},
    }
    graph = mocker.Mock()
    graph.invoke.return_value = state
    mocker.patch("mascan.orchestrator.graph.build_graph", return_value=graph)

    report = run("question", thread_id="thread-1")

    assert report.component_metrics == state["component_metrics"]
    assert report.metadata["execution"]["duration_seconds"] >= 0
    assert report.metadata["execution"]["duration_seconds"] != 12.0
    assert report.metadata["execution"]["token_usage"] == {
        "input_tokens": 300,
        "output_tokens": 70,
        "total_tokens": 370,
    }
