from mascan.contracts.planning import AgentAssignment
from mascan.orchestrator.planner import PLANNER_SYSTEM_PROMPT, _filter_to_known_agents


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


def test_planner_assigns_required_uploaded_evidence_to_an_agent() -> None:
    assert "evidence_documents" in PLANNER_SYSTEM_PROMPT
    assert "Copy its filename exactly from rag_search" in PLANNER_SYSTEM_PROMPT
    assert "at least one suitable agent must receive that filename" in PLANNER_SYSTEM_PROMPT


def test_assignment_defaults_to_no_uploaded_documents() -> None:
    assert assignment("economics", "Assess costs").evidence_documents == []
