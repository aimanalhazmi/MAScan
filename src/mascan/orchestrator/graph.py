from collections.abc import Callable, Iterator
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from mascan.agents.registry import agent_registry
from mascan.contracts import FinalReport
from mascan.contracts.validation import ValidationReport
from mascan.core.logging import configure_logging, get_logger
from mascan.orchestrator.adapters import make_agent_node
from mascan.orchestrator.planner import planner_node
from mascan.orchestrator.state import GraphState
from mascan.orchestrator.synthesizer import synthesizer_node
from mascan.orchestrator.validator import validator_node

MAX_INFO_REQUESTS = 3

logger = get_logger("orchestrator.graph")

compiled_graph: Any | None = None


def handle_info_request(state: GraphState) -> dict[str, Any]:
    """Pause the graph and ask the user the planner's clarification question.

    Uses LangGraph's interrupt so the question can surface over HTTP or the
    terminal; the run resumes once the answer arrives and loops back to the
    planner. Capped at MAX_INFO_REQUESTS to avoid looping forever.
    """
    if state.info_request is None:
        return {}

    counter = state.info_request_counter + 1
    if counter > MAX_INFO_REQUESTS:
        logger.warning("Info request counter exceeded threshold. Proceeding without answer.")
        return {"info_request": None, "info_request_counter": counter}

    question = state.info_request.question
    answer = interrupt(question)  # resumes with the user's answer

    updates: dict[str, Any] = {"info_request": None, "info_request_counter": counter}
    if answer:
        updates["user_input"] = _append_clarification_to_user_input(
            state.user_input, question, str(answer)
        )
    else:
        logger.warning("No clarification answer received. Planner will continue without new input.")
    return updates


def _append_clarification_to_user_input(
    user_input: str,
    question: str,
    answer: str,
) -> str:
    """Append Q/A context so the next planner pass sees user clarifications."""
    return (
        f"{user_input}\n\n"
        "Additional clarification provided during planning:\n"
        f"- Question: {question}\n"
        f"- Answer: {answer}"
    )


def agents_passthrough(state: GraphState) -> GraphState:
    """Passthrough node that routes to all agent nodes."""
    return state


def build_graph() -> Any:
    global compiled_graph
    if compiled_graph is not None:
        return compiled_graph

    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("handle_info_request", handle_info_request)
    graph.add_node("agents", agents_passthrough)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("validator", validator_node)

    agents = agent_registry.all()
    if not agents:
        raise RuntimeError(
            "No agents registered. Make sure to import the agent modules "
            "(e.g. `import mascan.agents.economics`) before building the graph."
        )

    for agent in agents:
        graph.add_node(agent.name, make_agent_node(agent))

    # Edges
    graph.add_edge(START, "planner")

    # Conditional routing from planner
    graph.add_conditional_edges(
        "planner",
        route_planner,
        {
            "handle_info_request": "handle_info_request",
            "agents": "agents",
        },
    )

    # Info request loops back to planner
    graph.add_edge("handle_info_request", "planner")

    # All agents route from agents passthrough and connect to synthesizer
    for agent in agents:
        graph.add_edge("agents", agent.name)
        graph.add_edge(agent.name, "synthesizer")

    graph.add_edge("synthesizer", "validator")
    graph.add_edge("validator", END)

    # Checkpointer lets the clarification interrupt pause and resume by thread_id.
    compiled_graph = graph.compile(checkpointer=MemorySaver())
    logger.info(f"Graph compiled with {len(agents)} agent node(s).")
    return compiled_graph


def run(
    query: str,
    thread_id: str | None = None,
    on_clarify: Callable[[str], str] | None = None,
) -> FinalReport:
    """Run end-to-end and return the FinalReport.

    If the planner asks for clarification, on_clarify is called with the
    question and must return the answer. Without it the run proceeds without
    clarification (headless callers can't prompt a user).
    """
    configure_logging()
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid4())}}
    result = graph.invoke(GraphState(user_input=query), config=config)
    while (question := interrupt_question(result)) is not None:
        answer = on_clarify(question) if on_clarify else ""
        result = graph.invoke(Command(resume=answer or ""), config=config)
    return state_to_report(result)


def stream(query: str, thread_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Stream node updates. Yields a `{"node": "__interrupt__", "question": ...}`
    event when the planner needs clarification; call resume() to continue."""
    configure_logging()
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid4())}}
    yield from _emit(graph.stream(GraphState(user_input=query), config=config, stream_mode="updates"))


def resume(thread_id: str, answer: str) -> Iterator[dict[str, Any]]:
    """Resume a stream paused by a clarification interrupt on this thread."""
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    yield from _emit(graph.stream(Command(resume=answer), config=config, stream_mode="updates"))


def _emit(chunks: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for chunk in chunks:
        for node_name, update in chunk.items():
            if node_name == "__interrupt__":
                yield {"node": "__interrupt__", "question": update[0].value}
            else:
                yield {"node": node_name, "update": update}


def interrupt_question(result: dict[str, Any]) -> str | None:
    """Return the pending clarification question, or None if not interrupted."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


def state_to_report(state_dict: dict[str, Any]) -> FinalReport:
    """Convert the graph's final state dict into a FinalReport."""
    validation = state_dict.get("validation")
    if isinstance(validation, ValidationReport):
        validation_payload = validation.model_dump(mode="json")
    elif isinstance(validation, dict):
        validation_payload = validation
    else:
        validation_payload = {}
    return FinalReport(
        user_input=state_dict.get("user_input", ""),
        summary=state_dict.get("final_summary", ""),
        rendered_markdown=state_dict.get("final_markdown", ""),
        plan=state_dict.get("plan", {}),
        agent_reports=state_dict.get("reports", {}),
        failures=state_dict.get("failures", {}),
        metadata={"validation": validation_payload},
    )

def route_planner(state: GraphState) -> str:
    """Route from planner based on whether it needs more info or can proceed to agents."""
    if state.info_request is not None:
        return "handle_info_request"
    return "agents"
