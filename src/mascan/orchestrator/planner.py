from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mascan.agents.registry import agent_registry
from mascan.contracts.planning import AgentAssignment, InformationRequest
from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger
from mascan.core.settings import get_settings
from mascan.orchestrator.state import GraphState

logger = get_logger("orchestrator.planner")

PLANNER_SYSTEM_PROMPT = """\
You are the planner of a PESTEL multi-agent market-analysis system.

Available agents (each specialises in one PESTEL dimension):
{available_agents}

Your job:
1. Read the user's question.
2. Decide wheter the user provided enough information to investigate the question.
   If not, return a clarification request.
3. Once the user provides sufficient information, decide which agents should investigate it.
   Only pick agents whose dimension is genuinely relevant.
   Skip agents whose dimension doesn't apply to this question.
4. For each selected agent, write:
   - an objective_context: a robust domain-specific brief for that agent.
     This is the only user-query context the agent will receive, so preserve
     every detail that matters for that agent's capabilities: entities,
     company details, product description, geography, time horizon, decision
     scope, constraints, assumptions, and what the agent should ignore.
     Tailor the objective_context to the selected agent's domain rather than
     copying the whole user question.
   - 1 to 3 short, specific sub-tasks describing exactly what that agent
     should investigate.

Return a JSON object with an "assignments" array. Each element has
"agent_name" (string), "objective_context" (string), and "tasks" (list of strings).
Agents you do NOT pick must NOT appear in the output.
Do not add facts that are not present in the user question or runtime context.
"""


class PlanModel(BaseModel):
    """Structured output the planner LLM is forced to return."""

    output: list[AgentAssignment] | InformationRequest = Field(
        description="List of agent-task assignments or a request for more information.",
    )


def planner_node(state: GraphState) -> dict[str, Any]:
    """LangGraph node: build the plan and write it to state."""
    available = agent_registry.all_names()
    if not available:
        logger.warning("No agents registered; planner has nothing to plan.")
        return {"plan": {}}

    settings = get_settings()
    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.0,  # planning should be deterministic   TODO: How should we handle information requests?
        max_tokens=1000,
    )
    structured_llm = llm.with_structured_output(PlanModel)

    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        available_agents="\n".join(f"- {name}" for name in available)
    )
    # Adding runtime context to planner agent
    runtime = state.runtime_context.model_dump()
    user_prompt = (
        "Runtime context:\n"
        f"- Current date: {runtime['current_date']}\n"
        f"- Timezone: {runtime['timezone']}\n\n"
        f"User question:\n{state.user_input}"
    )

    result: PlanModel = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    if isinstance(result.output, InformationRequest):
        logger.info(f"Planner requested more information: {result.output.question}")
        return {"plan": {}, "info_request": result.output}

    raw_plan = {a.agent_name: a for a in result.output if isinstance(a, AgentAssignment)}
    plan = _filter_to_known_agents(raw_plan, available)
    logger.info(f"Planner selected {len(plan)} agent(s): {sorted(plan.keys())}")
    return {"plan": plan}


def _filter_to_known_agents(
    plan: dict[str, AgentAssignment], available: list[str]
) -> dict[str, AgentAssignment]:
    known = set(available)
    filtered: dict[str, AgentAssignment] = {}
    for name, assignment in plan.items():
        if name not in known:
            logger.warning(f"Planner hallucinated unknown agent {name}; dropping.")
            continue
        if not assignment.tasks:
            continue
        filtered[name] = assignment
    return filtered
