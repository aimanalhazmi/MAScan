import mascan.agents.economics  # noqa: F401
import mascan.agents.environmental  # noqa: F401
import mascan.agents.legal  # noqa: F401
import mascan.agents.political  # noqa: F401
import mascan.agents.social  # noqa: F401
import mascan.agents.technological  # noqa: F401
from mascan.orchestrator.graph import build_graph


def test_validator_runs_after_synthesizer_and_before_end() -> None:
    graph = build_graph().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ("synthesizer", "validator") in edges
    assert ("validator", "__end__") in edges
    assert ("synthesizer", "__end__") not in edges
