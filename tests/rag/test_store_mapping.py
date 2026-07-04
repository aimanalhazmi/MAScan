"""The Chunk <-> Document mapping must preserve source + citation (the attribution path)."""

from mascan.contracts.retrieval import Chunk, Citation
from mascan.rag.store import chunk_to_document, document_to_retrieved_chunk


def test_round_trip_preserves_citation_and_metadata():
    chunk = Chunk(
        content="Revenue rose 12%.",
        source="upload",
        citation=Citation(document="10k.pdf", page=4, block="table-2"),
        metadata={"ticker": "ACME"},
    )

    rc = document_to_retrieved_chunk(chunk_to_document(chunk), score=0.87)

    assert rc.content == chunk.content
    assert rc.source == "upload"
    assert rc.citation == chunk.citation
    assert rc.metadata["ticker"] == "ACME"
    assert rc.score == 0.87
