from mascan.contracts.planning import AgentAssignment
from mascan.orchestrator.planner import _filter_to_known_agents


def assignment(name: str, task: str) -> AgentAssignment:
    return AgentAssignment(
        agent_name=name,
        objective_context="Assess European market conditions.",
        tasks=[task],
    )


def test_economic_alias_is_normalized_to_registered_economics_agent() -> None:
    plan = _filter_to_known_agents(
        {"economic": assignment("economic", "Assess energy costs")},
        ["political", "economics", "social"],
    )

    assert list(plan) == ["economics"]
    assert plan["economics"].agent_name == "economics"
    assert plan["economics"].tasks == ["Assess energy costs"]


def test_alias_and_canonical_duplicates_keep_first_assignment() -> None:
    plan = _filter_to_known_agents(
        {
            "economic": assignment("economic", "First task"),
            "economics": assignment("economics", "Duplicate task"),
        },
        ["economics"],
    )

    assert plan["economics"].tasks == ["First task"]
