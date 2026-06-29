from collections.abc import Iterator
import sys
from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain.agents.middleware import HumanInTheLoopMiddleware

from mascan.agents.registry import agent_registry
from mascan.contracts import FinalReport
from mascan.core.logging import configure_logging, get_logger
from mascan.orchestrator.adapters import make_agent_node
from mascan.orchestrator.planner import planner_node
from mascan.orchestrator.state import GraphState
from mascan.orchestrator.synthesizer import synthesizer_node

MAX_INFO_REQUESTS = 3

logger = get_logger("orchestrator.graph")

compiled_graph: Any | None = None


def _handle_info_request(state: GraphState) -> GraphState:
    """This node handles information requests from the planner by asking the user for clarification.
    After collecting the user's response, it loops back to the planner with the updated context.
    Maximum of 3 info requests are allowed to prevent infinite loops.
    """

    if state.info_request is None:
        return state

    logger.info(f"Planner requested info: {state.info_request.question}")

    # Increment counter to prevent infinite loops
    state.info_request_counter += 1
    if state.info_request_counter > MAX_INFO_REQUESTS:
        logger.warning("Info request counter exceeded threshold. Proceeding without answer.")
        state.info_request = None
        return state

    answer = _prompt_for_console_input(state.info_request.question)
    if answer:
        state.user_input = _append_clarification_to_user_input(
            state.user_input,
            state.info_request.question,
            answer,
        )
    else:
        logger.warning("No clarification answer received. Planner will continue without new input.")

    state.info_request = None
    return state


def _prompt_for_console_input(question: str) -> str | None:
    """Collect a clarification answer from stdin when running in a terminal."""
    if not sys.stdin.isatty():
        logger.warning("stdin is not interactive; skipping clarification prompt.")
        return None

    print("\n[MAScan needs one clarification before continuing]")
    print(question)
    print("Type your answer and press Enter (or press Ctrl+C to skip):")

    try:
        while True:
            answer = input("> ").strip()
            if answer:
                return answer
            print("Please enter a non-empty answer.")
    except KeyboardInterrupt:
        print("\nSkipping clarification.")
        return None


def _append_clarification_to_user_input(
    user_input: str,
    question: str,
    answer: str,
) -> str:
    """Append Q/A context so the next planner pass sees user clarifications."""
    return (
        f"{user_input}\n\n"
        "Additional clarification provided during orchestration:\n"
        f"- Question: {question}\n"
        f"- Answer: {answer}"
    )


def _agents_passthrough(state: GraphState) -> GraphState:
    """Passthrough node that routes to all agent nodes."""
    return state


def build_graph() -> Any:
    global compiled_graph
    if compiled_graph is not None:
        return compiled_graph

    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("handle_info_request", _handle_info_request)
    graph.add_node("agents", _agents_passthrough)
    graph.add_node("synthesizer", synthesizer_node)

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
    
    graph.add_edge("synthesizer", END)

    compiled_graph = graph.compile()
    logger.info(f"Graph compiled with {len(agents)} agent node(s).")
    return compiled_graph


def run(query: str) -> FinalReport:
    """Run the orchestrator end-to-end and return the FinalReport."""
    configure_logging()
    graph = build_graph()
    initial = GraphState(user_input=query)
    final_state_dict: dict[str, Any] = graph.invoke(initial)
    return state_to_report(final_state_dict)


def stream(query: str) -> Iterator[dict[str, Any]]:
    configure_logging()
    graph = build_graph()
    initial = GraphState(user_input=query)
    for chunk in graph.stream(initial, stream_mode="updates"):
        for node_name, update in chunk.items():
            yield {"node": node_name, "update": update}


def state_to_report(state_dict: dict[str, Any]) -> FinalReport:
    """Convert the graph's final state dict into a FinalReport."""
    return FinalReport(
        user_input=state_dict.get("user_input", ""),
        summary=state_dict.get("final_summary", ""),
        rendered_markdown=state_dict.get("final_markdown", ""),
        plan=state_dict.get("plan", {}),
        agent_reports=state_dict.get("reports", {}),
        failures=state_dict.get("failures", {}),
    )

def route_planner(state: GraphState) -> str:
    """Route from planner based on whether it needs more info or can proceed to agents."""
    if state.info_request is not None:
        return "handle_info_request"
    return "agents"