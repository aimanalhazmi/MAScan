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

CLARIFY_SYSTEM_PROMPT = """\
You are the planner of a PESTEL multi-agent market-analysis system.

Before any analysis begins, decide whether you understand the user's intent
well enough to plan confidently. A request is clear when the goal, the market
or entity, and the scope are all evident.

- If the request is clear, set needs_clarification to false.
- Otherwise set needs_clarification to true and ask exactly one concise
  clarifying question that resolves the most important ambiguity, such as the
  goal, the scope, the market or entity, the geography, or the time horizon.

Lean towards confirming intent, but do not ask when the request is already
specific enough to act on.
"""


class PlanModel(BaseModel):
    """Structured output the planner LLM is forced to return."""

    assignments: list[AgentAssignment] | InformationRequest = Field(
        description="List of agent-task assignments or a request for more information.",
    )


class IntentCheck(BaseModel):
    """Structured output of the pre-planning intent confirmation step."""

    needs_clarification: bool = Field(
        description="True if the request is too ambiguous to plan confidently.",
    )
    question: str = Field(
        default="",
        description="The clarifying question to ask, when needs_clarification is true.",
    )


def planner_node(state: GraphState) -> dict[str, Any]:
    """LangGraph node: build the plan and write it to state."""
    available = agent_registry.all_names()
    if not available:
        logger.warning("No agents registered; planner has nothing to plan.")
        return {"plan": {}}

    settings = get_settings()
    runtime = state.runtime_context.model_dump()
    user_prompt = build_user_prompt(state, runtime)

    # Confirm the user's intent once before planning, unless the request is
    # already specific enough to act on.
    if state.info_request_counter == 0:
        question = clarify_intent(user_prompt, settings)
        if question:
            logger.info(f"Planner requesting intent clarification: {question}")
            return {"plan": {}, "info_request": InformationRequest(question=question)}

    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.0,
        max_tokens=1000,
    )
    structured_llm = llm.with_structured_output(PlanModel)
    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        available_agents="\n".join(f"- {name}" for name in available)
    )

    result: PlanModel = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    if isinstance(result.assignments, InformationRequest):
        logger.info(f"Planner requested more information: {result.assignments.question}")
        return {"plan": {}, "info_request": result.assignments}

    raw_plan = {a.agent_name: a for a in result.assignments if isinstance(a, AgentAssignment)}
    plan = _filter_to_known_agents(raw_plan, available)
    logger.info(f"Planner selected {len(plan)} agent(s): {sorted(plan.keys())}")
    return {"plan": plan}


def build_user_prompt(state: GraphState, runtime: dict[str, Any]) -> str:
    """Compose the planner's user message from runtime context and the question."""
    return (
        "Runtime context:\n"
        f"- Current date: {runtime['current_date']}\n"
        f"- Timezone: {runtime['timezone']}\n\n"
        f"User question:\n{state.user_input}"
    )


def clarify_intent(user_prompt: str, settings: Any) -> str | None:
    """Return one clarifying question, or None if the request is clear enough."""
    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.0,
        max_tokens=200,
    )
    structured_llm = llm.with_structured_output(IntentCheck)
    result: IntentCheck = structured_llm.invoke([
        SystemMessage(content=CLARIFY_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    question = result.question.strip()
    return question if result.needs_clarification and question else None


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
