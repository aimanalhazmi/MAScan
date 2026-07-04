"""Context formatting must number docs [1..n] and surface structured citation tags."""

from mascan.contracts.retrieval import Citation, RetrievedChunk
from mascan.rag.answer import citation_tag, format_context


def testcitation_tag_includes_page_and_block():
    tag = citation_tag(Citation(document="10k.pdf", page=4, block="table-2"))
    assert tag == "10k.pdf p.4 (table-2)"
    assert citation_tag(Citation(document="news.html")) == "news.html"


def testformat_context_numbers_from_one():
    chunks = [
        RetrievedChunk(content="A", source="s", citation=Citation(document="d1")),
        RetrievedChunk(content="B", source="s", citation=Citation(document="d2")),
    ]
    ctx = format_context(chunks)
    assert ctx.startswith("[1] d1\nA")
    assert "[2] d2\nB" in ctx
