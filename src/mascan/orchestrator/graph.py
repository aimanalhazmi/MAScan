from collections.abc import Iterator
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

logger = get_logger("orchestrator.graph")

compiled_graph: Any | None = None


def build_graph() -> Any:
    global compiled_graph
    if compiled_graph is not None:
        return compiled_graph

    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("synthesizer", synthesizer_node)


    agents = agent_registry.all()
    if not agents:
        raise RuntimeError(
            "No agents registered. Make sure to import the agent modules "
            "(e.g. `import mascan.agents.economics`) before building the graph."
        )

    for agent in agents:
        graph.add_node(agent.name, make_agent_node(agent))

    # Edges: planner -> every agent -> synthesizer
    graph.add_edge(START, "planner")
    for agent in agents:
        graph.add_edge("planner", agent.name)
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