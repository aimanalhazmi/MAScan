import mascan.rag.correction as correction
import mascan.rag.query_gen as query_gen
import mascan.rag.rerank as rerank_mod
from mascan.contracts.retrieval import Citation, RetrievalQuery, RetrievedChunk
from mascan.rag.retriever import (
    AdaptiveRetriever,
    CorrectiveRetriever,
    DecomposingRetriever,
    MultiHydeRetriever,
    RerankingRetriever,
    merge,
)


def _chunk(content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        content=content, source="news_api", citation=Citation(document="d"), score=score
    )


class _FakeBase:
    """Records the queries it sees; returns one chunk per query."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.seen.append((query.query, query.k))
        return [_chunk(f"{query.query}-doc")]


def test_merge_dedupes_keeping_highest_score():
    merged = merge([_chunk("a", 0.1), _chunk("b", 0.5), _chunk("a", 0.9)])
    assert {c.content for c in merged} == {"a", "b"}
    assert {c.content: c.score for c in merged}["a"] == 0.9


async def test_multihyde_fans_out_over_variants(monkeypatch):
    async def fake_hyde(_question: str) -> list[str]:
        return ["v1", "v2"]

    monkeypatch.setattr(query_gen, "generate_hyde_documents", fake_hyde)
    base = _FakeBase()
    out = await MultiHydeRetriever(base).retrieve(RetrievalQuery(query="Q"))

    assert [q for q, _ in base.seen] == ["Q", "v1", "v2"]  # original + 2 variants
    assert {c.content for c in out} == {"Q-doc", "v1-doc", "v2-doc"}


async def test_decompose_fans_out_over_subqueries(monkeypatch):
    async def fake_decompose(_question: str) -> list[str]:
        return ["s1", "s2"]

    monkeypatch.setattr(query_gen, "decompose_query", fake_decompose)
    base = _FakeBase()
    out = await DecomposingRetriever(base).retrieve(RetrievalQuery(query="Q"))

    assert [q for q, _ in base.seen] == ["s1", "s2"]
    assert {c.content for c in out} == {"s1-doc", "s2-doc"}


async def test_reranking_overfetches_then_trims(monkeypatch):
    async def fake_rerank(_q, chunks, top_k):
        return chunks[:top_k]

    monkeypatch.setattr(rerank_mod, "rerank", fake_rerank)
    base = _FakeBase()
    out = await RerankingRetriever(base, fetch_k=10).retrieve(RetrievalQuery(query="Q", k=1))

    assert base.seen == [("Q", 10)]  # over-fetched with fetch_k, not user k
    assert len(out) == 1


async def test_corrective_retries_with_rewrite_when_results_weak(monkeypatch):
    grades = iter([False, True])  # first attempt weak, second good

    async def fake_grade(_q, _chunks):
        return next(grades)

    async def fake_rewrite(_q):
        return "better query"

    monkeypatch.setattr(correction, "grade_relevance", fake_grade)
    monkeypatch.setattr(correction, "rewrite_query", fake_rewrite)
    base = _FakeBase()
    out = await CorrectiveRetriever(base).retrieve(RetrievalQuery(query="Q"))

    assert [q for q, _ in base.seen] == ["Q", "better query"]  # retried with rewrite
    assert {c.content for c in out} == {"better query-doc"}


async def test_corrective_no_retry_when_results_good(monkeypatch):
    async def fake_grade(_q, _chunks):
        return True

    monkeypatch.setattr(correction, "grade_relevance", fake_grade)
    base = _FakeBase()
    await CorrectiveRetriever(base).retrieve(RetrievalQuery(query="Q"))

    assert [q for q, _ in base.seen] == ["Q"]  # graded good, no rewrite/retry


async def test_adaptive_stays_cheap_when_first_pass_is_good(monkeypatch):
    async def fake_grade(_q, _chunks):
        return True

    monkeypatch.setattr(correction, "grade_relevance", fake_grade)
    cheap, full = _FakeBase(), _FakeBase()
    out = await AdaptiveRetriever(cheap, full).retrieve(RetrievalQuery(query="Q"))

    assert full.seen == []  # never escalated
    assert {c.content for c in out} == {"Q-doc"}


async def test_adaptive_escalates_to_full_when_first_pass_is_weak(monkeypatch):
    async def fake_grade(_q, _chunks):
        return False

    monkeypatch.setattr(correction, "grade_relevance", fake_grade)
    cheap, full = _FakeBase(), _FakeBase()
    out = await AdaptiveRetriever(cheap, full).retrieve(RetrievalQuery(query="Q"))

    assert full.seen == [("Q", 5)]  # escalated with the original query
    assert {c.content for c in out} == {"Q-doc"}
