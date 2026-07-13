"""Retriever protocol and implementations.

`RetrieverProtocol` is the interface everything else depends on. `NullRetriever`
returns nothing and is the default when RAG is off. `PGVectorRetriever` does the
real dense search.
"""

import asyncio
from typing import Protocol, runtime_checkable

from mascan.contracts.retrieval import RetrievalQuery, RetrievedChunk
from mascan.rag.store import document_to_retrieved_chunk, get_vector_store


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Anything that turns a query into ranked chunks."""

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]: ...


class NullRetriever:
    """Returns nothing. Used when no vector store is configured."""

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        return []


class PGVectorRetriever:
    """Dense similarity search over the PGVector store."""

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        store = get_vector_store()


        embedding = await store.embeddings.aembed_query(query.query)
        results = await store.asimilarity_search_with_score_by_vector(embedding, k=query.k)
        return [document_to_retrieved_chunk(doc, score) for doc, score in results]


# How many candidates to fetch per query before reranking. Over-fetch wide so
# the reranker actually has the right chunk to promote; it trims back to query.k.
DENSE_FETCH_K = 10


def merge(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """De-duplicate by content, keeping the highest score; preserve first-seen order."""
    best: dict[str, RetrievedChunk] = {}
    for c in chunks:
        prev = best.get(c.content)
        if prev is None or c.score > prev.score:
            best[c.content] = c
    return list(best.values())


class MultiHydeRetriever:
    """Multi-HyDE: expand the query into several hypothetical passages plus the
    original, retrieve for each, and merge the results."""

    def __init__(self, base: RetrieverProtocol) -> None:
        self._base = base

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        from mascan.rag.query_gen import generate_hyde_documents

        variants = [query.query, *await generate_hyde_documents(query.query)]

        batches = await asyncio.gather(
            *(self._base.retrieve(query.model_copy(update={"query": v})) for v in variants)
        )
        return merge([c for batch in batches for c in batch])


class DecomposingRetriever:
    """Split a multi-hop question into sub-questions, retrieve for each, and merge."""

    def __init__(self, base: RetrieverProtocol) -> None:
        self._base = base

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        from mascan.rag.query_gen import decompose_query

        subs = await decompose_query(query.query)
        batches = await asyncio.gather(
            *(self._base.retrieve(query.model_copy(update={"query": sub})) for sub in subs)
        )
        return merge([c for batch in batches for c in batch])


class RerankingRetriever:
    """Over-fetch candidates, then LLM-rerank down to query.k."""

    def __init__(self, base: RetrieverProtocol, fetch_k: int = DENSE_FETCH_K) -> None:
        self._base = base
        self._fetch_k = fetch_k

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        from mascan.rag.rerank import rerank

        candidates = await self._base.retrieve(query.model_copy(update={"k": self._fetch_k}))
        return await rerank(query.query, candidates, query.k)


class CorrectiveRetriever:
    """CRAG: grade the results against the question; if weak, rewrite the query
    and retry, up to max_retries."""

    def __init__(self, base: RetrieverProtocol, max_retries: int = 1) -> None:
        self._base = base
        self._max_retries = max_retries

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        from mascan.rag.correction import grade_relevance, rewrite_query

        current = query
        chunks = await self._base.retrieve(current)
        for _ in range(self._max_retries):
            if await grade_relevance(query.query, chunks):
                break
            current = query.model_copy(update={"query": await rewrite_query(current.query)})
            chunks = await self._base.retrieve(current)
        return chunks


class AdaptiveRetriever:
    """Run a cheap retriever first; only fall back to the expensive `full`
    pipeline when the cheap result is weak."""

    def __init__(self, cheap: RetrieverProtocol, full: RetrieverProtocol) -> None:
        self._cheap = cheap
        self._full = full

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        from mascan.rag.correction import grade_relevance

        results = await self._cheap.retrieve(query)
        if await grade_relevance(query.query, results):
            return results
        return await self._full.retrieve(query)


def get_retriever() -> RetrieverProtocol:
    """Build the retrieval pipeline: cheap dense+rerank first, escalating to the
    full decompose→HyDE→rerank→CRAG path when needed. Returns NullRetriever when
    RAG is disabled.
    """
    from mascan.core.settings import get_settings

    settings = get_settings()
    if not settings.database_url:
        return NullRetriever()

    base = PGVectorRetriever()
    cheap = RerankingRetriever(base)
    full: RetrieverProtocol = MultiHydeRetriever(base)
    full = DecomposingRetriever(full)
    full = RerankingRetriever(full)
    full = CorrectiveRetriever(full, max_retries=settings.rag_max_retries)
    return AdaptiveRetriever(cheap, full)
