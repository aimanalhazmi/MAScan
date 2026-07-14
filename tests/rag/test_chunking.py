from mascan.contracts.retrieval import Citation
from mascan.rag.chunking import CHUNK_SIZE, chunk_text


def test_long_text_splits_and_carries_citation():
    text = "Intro paragraph.\n\n" + ("word " * 1000)  # well over one chunk
    cite = Citation(document="Reuters: Fed holds rates")
    chunks = chunk_text(text, source="news_api", citation=cite)

    assert len(chunks) > 1  # actually split
    assert all(c.citation.document == "Reuters: Fed holds rates" for c in chunks)
    assert all(c.source == "news_api" for c in chunks)
    assert all(len(c.content) <= CHUNK_SIZE for c in chunks)


def test_short_text_is_single_chunk():
    chunks = chunk_text("tiny", source="news_api", citation=Citation(document="d"))
    assert len(chunks) == 1
    assert chunks[0].content == "tiny"
