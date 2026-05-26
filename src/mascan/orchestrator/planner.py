from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mascan.agents.registry import agent_registry
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
2. Decide which agents should investigate it. Only pick agents whose
   dimension is genuinely relevant. Skip agents whose dimension doesn't
   apply to this question.
3. For each selected agent, write 1 to 3 short, specific sub-tasks
   describing exactly what that agent should investigate.

Return a JSON object with an "assignments" array. Each element has
"agent_name" (string) and "tasks" (list of strings).
Agents you do NOT pick must NOT appear in the output.
"""


class AgentTasks(BaseModel):
    """Tasks assigned to a single agent."""

    agent_name: str = Field(description="Name of the agent.")
    tasks: list[str] = Field(description="List of sub-tasks for this agent.")


class PlanModel(BaseModel):
    """Structured output the planner LLM is forced to return."""

    assignments: list[AgentTasks] = Field(
        description="List of agent-task assignments.",
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
        temperature=0.0,  # planning should be deterministic
        max_tokens=1000,
    )
    structured_llm = llm.with_structured_output(PlanModel)

    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        available_agents="\n".join(f"- {name}" for name in available)
    )

    result: PlanModel = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.user_input),
    ])

    raw_plan = {a.agent_name: a.tasks for a in result.assignments}
    plan = _filter_to_known_agents(raw_plan, available)
    logger.info("Planner selected %d agent(s): %s", len(plan), sorted(plan.keys()))
    return {"plan": plan}


def _filter_to_known_agents(
    plan: dict[str, list[str]], available: list[str]
) -> dict[str, list[str]]:
    known = set(available)
    filtered: dict[str, list[str]] = {}
    for name, tasks in plan.items():
        if name not in known:
            logger.warning("Planner hallucinated unknown agent %r; dropping.", name)
            continue
        if not tasks:
            continue
        filtered[name] = tasks
    return filtered