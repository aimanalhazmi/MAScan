"""rag_search — lets an agent query MAScan's internal RAG index.."""

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, Field

from mascan.contracts.retrieval import RetrievalQuery
from mascan.contracts.tools import ToolResult
from mascan.rag.retriever import get_retriever
from mascan.tools.base import BaseTool

_T = TypeVar("_T")


def run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine from sync code, whether or not a loop is already running.

    Agents may be invoked from a plain sync context (no loop → asyncio.run) or
    from inside FastAPI's async endpoints (a loop is already running on this
    thread → asyncio.run would raise, so run it in a worker thread instead).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class RagSearchInput(BaseModel):
    query: str = Field(..., description="Question to search the internal knowledge base for.")
    k: int = Field(5, ge=1, le=20, description="How many passages to return.")


class RagSearchTool(BaseTool):
    name = "rag_search"
    description = (
        "Search MAScan's internal knowledge base (ingested filings, reports, and "
        "articles) and return relevant passages, each with its source citation."
    )
    input_schema: ClassVar[type[BaseModel] | None] = RagSearchInput

    def run(self, query: str, k: int = 5, **_: Any) -> ToolResult[list[dict[str, Any]]]:
        try:
            chunks = run_async(get_retriever().retrieve(RetrievalQuery(query=query, k=k)))
            return ToolResult(
                success=True,
                data=[
                    {
                        "content": c.content,
                        "citation": c.citation.model_dump(),
                        "score": c.score,
                    }
                    for c in chunks
                ],
                source="rag_search",
                metadata={"query": query, "count": len(chunks)},
            )
        except Exception as exc:
            self.logger.exception("rag_search failed for query=%r", query)
            return ToolResult(success=False, source="rag_search", error=str(exc))
