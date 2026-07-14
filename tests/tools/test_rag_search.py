from typing import Any

from mascan.contracts.retrieval import Citation, RetrievedChunk
from mascan.tools.common.rag_search import RagSearchTool


def chunk(score: float, document: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=f"passage from {document}",
        source="upload",
        citation=Citation(document=document),
        score=score,
    )


def test_run_drops_passages_below_the_score_floor(mocker: Any) -> None:
    """A weak nearest neighbour must not reach the planner and pollute the plan."""
    mocker.patch(
        "mascan.tools.common.rag_search.run_sync",
        return_value=[chunk(0.72, "relevant.pdf"), chunk(0.41, "unrelated.pdf")],
    )
    mocker.patch("mascan.tools.common.rag_search.get_settings").return_value.rag_min_score = 0.5

    result = RagSearchTool().run(query="anything")

    assert result.success
    assert [c["citation"]["document"] for c in result.data] == ["relevant.pdf"]
    assert result.metadata == {"query": "anything", "count": 1, "retrieved": 2}
