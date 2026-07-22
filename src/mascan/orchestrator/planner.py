import json
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from mascan.agents.registry import agent_registry
from mascan.contracts.planning import AgentAssignment, InformationRequest, IntentCheck, PlanModel
from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger
from mascan.core.metrics import measure_component
from mascan.core.settings import get_settings
from mascan.orchestrator.state import GraphState
from mascan.tools.registry import tool_registry

logger = get_logger("orchestrator.planner")

# Registered agent names for the six PESTEL dimensions.
PESTEL_AGENTS: frozenset[str] = frozenset(
    {"political", "economics", "social", "technological", "environmental", "legal"}
)

# Common planner misspellings / singular forms → registered agent names.
AGENT_NAME_ALIASES: dict[str, str] = {
    "economic": "economics",
    "economic agent": "economics",
    "economics agent": "economics",
    "economy": "economics",
    "environment": "environmental",
    "politics": "political",
    "technology": "technological",
    "tech": "technological",
}

PLANNER_SYSTEM_PROMPT = """\
You are the planner of a PESTEL multi-agent market-analysis system.

Available agents (each specialises in one PESTEL dimension):
{available_agents}

You may already have searched MAScan's knowledge base, which holds the documents the
user uploaded, such as company filings and product briefs. Any passages it returned
are in this conversation as rag_search results.

Your job:
1. Read the user's question.
2. Decide wheter the user provided enough information to investigate the question.
   If not, return a clarification request.
3. Once the user provides sufficient information, decide which agents should investigate it.
   Only pick agents whose dimension is genuinely relevant.
   Skip agents whose dimension doesn't apply to this question.
   Exception: when the user explicitly asks for a PESTEL analysis or lists the
   PESTEL headings (Political, Economic, Social, Technological, Environmental,
   Legal), treat ALL SIX dimensions as relevant and assign all six agents.
4. For each selected agent, write:
   - an objective_context: a robust domain-specific brief for that agent.
     Preserve every detail that matters for that agent's capabilities: entities,
     company details, product description, geography, time horizon, decision
     scope, constraints, assumptions, and what the agent should ignore.
     Tailor the objective_context to the selected agent's domain rather than
     copying the whole user question.
   - 1 to 3 short, specific sub-tasks describing exactly what that agent
     should investigate.
   - evidence_documents: a list of exact uploaded-document filenames from the
     rag_search results that this agent needs. Use an empty list when none apply.
   - salient_factors: 3 to 6 of the MOST decision-relevant factors that this
     specific subject, industry, era, and geography demand in that agent's
     dimension. Name concrete factors, not generic category labels: e.g. for a
     2019 electric-vehicle maker's Economic dimension prefer "battery pack cost
     per kWh decline" and "EV purchase-incentive phase-out" over "market growth"
     or "consumer spending". These are coverage targets the agent must
     investigate and verify with evidence.

Return a JSON object with an "assignments" array. Each element has
"agent_name" (string), "objective_context" (string), "tasks" (list of strings),
"evidence_documents" (list of exact filenames), and "salient_factors"
(list of strings).
Agents you do NOT pick must NOT appear in the output.
Do not add unsupported facts to objective_context or tasks: assert nothing about
the case that is not in the user question, the runtime context, or what rag_search
returned. salient_factors are the exception — they are candidate factors to
investigate (hypotheses), so naming well-known factors for the named subject is
expected and encouraged; the agent will confirm or refute each with evidence.

Treat what rag_search returns as the ground truth about the user's company,
product, or market, and carry the parts that matter into the objective_context
of every agent that needs them. Agents cannot search the knowledge base
themselves, so anything you leave out is lost to them.

Assign each retrieved document only to agents whose tasks need it. Copy its
filename exactly from rag_search into evidence_documents; do not reconstruct or
shorten it. If the user explicitly requires an attached or uploaded document,
at least one suitable agent must receive that filename and a task requiring the
agent to use and cite its relevant evidence.
"""

LOOKUP_SYSTEM_PROMPT = """\
You are the planner of a PESTEL multi-agent market-analysis system.

Before planning, decide whether MAScan's knowledge base is worth searching. It holds
only the documents the user uploaded, such as company filings, product briefs, and
strategy papers. It holds no market news, regulations, or external research: the
agents gather all of that themselves.

- Call rag_search when the request names a company, product, or market the user is
  likely to have documented, using a short query naming that entity, once per entity.
- If the user says a document is attached, uploaded, or should be used, you MUST call
  rag_search for the named company or document topic.
- Call nothing when the request is self-contained, or when it asks about the wider
  world rather than about the user's own business.
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



def planner_node(state: GraphState) -> dict[str, Any]:
    """Measure one planner execution and return its state update."""
    return measure_component("planner", lambda: _planner_node(state))


def _planner_node(state: GraphState) -> dict[str, Any]:
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

    # The planner looks into the knowledge base first, then plans with whatever
    # it found. Both steps use the same model.
    llm = get_chat_model(
        model=settings.openai_model_default,
        temperature=0.0,
        max_tokens=2000,
    )
    knowledge_messages, rag_evidence = search_knowledge_base(llm, user_prompt)
    messages: list[Any] = [
        SystemMessage(
            content=PLANNER_SYSTEM_PROMPT.format(
                available_agents="\n".join(f"- {name}" for name in available)
            )
        ),
        HumanMessage(content=user_prompt),
        *knowledge_messages,
    ]

    result = cast(PlanModel, llm.with_structured_output(PlanModel).invoke(messages))

    if isinstance(result.assignments, InformationRequest):
        logger.info(f"Planner requested more information: {result.assignments.question}")
        return {"plan": {}, "info_request": result.assignments}

    raw_plan = {a.agent_name: a for a in result.assignments if isinstance(a, AgentAssignment)}
    plan = _filter_to_known_agents(raw_plan, available)
    logger.info(f"Planner selected {len(plan)} agent(s): {sorted(plan.keys())}")
    _warn_on_incomplete_pestel_coverage(state.user_input, plan, available)
    return {"plan": plan, "rag_evidence": rag_evidence}


def looks_like_pestel_request(user_input: str) -> bool:
    """True when the request explicitly asks for a full PESTEL analysis."""
    lowered = user_input.lower()
    if "pestel" in lowered:
        return True
    heading_terms = (
        "political",
        "economic",
        "social",
        "technological",
        "environmental",
        "legal",
    )
    return sum(1 for term in heading_terms if term in lowered) >= 4


def _warn_on_incomplete_pestel_coverage(
    user_input: str,
    plan: dict[str, AgentAssignment],
    available: list[str],
) -> None:
    """Log when a PESTEL request leaves one of the six dimensions unassigned."""
    if not looks_like_pestel_request(user_input):
        return
    expected = PESTEL_AGENTS & set(available)
    missing = sorted(expected - set(plan))
    if missing:
        logger.warning(
            "PESTEL request but planner omitted agent(s) %s; those categories "
            "will be uncovered in the final report.",
            missing,
        )


def search_knowledge_base(
    llm: BaseChatModel,
    user_prompt: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Let the planner search the uploaded documents, and return the exchange as messages."""
    if "rag_search" not in tool_registry.all_names():
        return [], []

    tool = tool_registry.get("rag_search")
    decision = llm.bind_tools([tool.as_langchain_tool()]).invoke([
        SystemMessage(content=LOOKUP_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    calls = getattr(decision, "tool_calls", None) or []
    if not calls:
        logger.info("Planner found no reason to search the knowledge base.")
        return [], []

    answers: list[Any] = []
    evidence: list[dict[str, Any]] = []
    seen_evidence: set[tuple[str, int | None, str]] = set()
    for call in calls:
        call_args = dict(call["args"])
        call_args["k"] = max(int(call_args.get("k") or 5), 10)
        result = tool.run(**call_args)
        if not result.success:
            logger.warning(f"rag_search failed during planning: {result.error}")
        passages = result.data if result.success and result.data else []
        logger.info(f"Planner searched {call['args'].get('query')!r}: {len(passages)} passage(s).")
        answers.append(ToolMessage(content=json.dumps(passages), tool_call_id=call["id"]))
        for passage in passages:
            if not isinstance(passage, dict):
                continue
            citation = passage.get("citation") or {}
            document = str(citation.get("document") or "uploaded document")
            page = citation.get("page") if isinstance(citation.get("page"), int) else None
            content = str(passage.get("content") or "").strip()
            key = (document, page, content)
            if content and key not in seen_evidence:
                seen_evidence.add(key)
                evidence.append(passage)

    return [decision, *answers], evidence


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
    result = cast(IntentCheck, structured_llm.invoke([
        SystemMessage(content=CLARIFY_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]))
    question = result.question.strip()
    return question if result.needs_clarification and question else None


def _normalize_agent_name(name: str, available: list[str]) -> str:
    normalized = name.strip().lower()
    canonical_names = {candidate.lower(): candidate for candidate in available}
    if normalized in canonical_names:
        return canonical_names[normalized]
    alias = AGENT_NAME_ALIASES.get(normalized)
    if alias and alias in available:
        logger.info("Planner alias %r mapped to registered agent %r.", name, alias)
        return alias
    return name


def _filter_to_known_agents(
    plan: dict[str, AgentAssignment], available: list[str]
) -> dict[str, AgentAssignment]:
    known = set(available)
    filtered: dict[str, AgentAssignment] = {}
    for name, assignment in plan.items():
        normalized = _normalize_agent_name(name, available)
        if normalized not in known:
            logger.warning(f"Planner hallucinated unknown agent {name}; dropping.")
            continue
        if not assignment.tasks:
            continue
        if normalized in filtered:
            logger.warning(
                "Planner returned duplicate assignments for %s; keeping the first.",
                normalized,
            )
            continue
        if normalized != name:
            assignment = assignment.model_copy(update={"agent_name": normalized})
        filtered[normalized] = assignment
    return filtered
