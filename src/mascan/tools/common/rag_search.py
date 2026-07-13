"""rag_search — lets an agent query MAScan's internal RAG index.."""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mascan.contracts.retrieval import RetrievalQuery
from mascan.contracts.tools import ToolResult
from mascan.core.settings import get_settings
from mascan.rag.retriever import get_retriever
from mascan.tools.base import BaseTool

background_loop: asyncio.AbstractEventLoop | None = None
loop_lock = threading.Lock()


def get_background_loop() -> asyncio.AbstractEventLoop:
    """One long-lived event loop for every sync caller of the async RAG stack.

    The vector store's database engine and the embedding client bind to the loop
    that first uses them, and fail with a connection error when a later call
    reaches them from a different loop. Keeping a single loop for all bridged
    calls also stops each call from opening its own connection pool.
    """
    global background_loop
    with loop_lock:
        if background_loop is None:
            background_loop = asyncio.new_event_loop()
            threading.Thread(
                target=background_loop.run_forever, name="rag-loop", daemon=True
            ).start()
        return background_loop


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code (agents, the planner) and wait for it."""
    return asyncio.run_coroutine_threadsafe(coro, get_background_loop()).result()


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
            floor = get_settings().rag_min_score
            relevant = [c for c in chunks if c.score >= floor]
            return ToolResult(
                success=True,
                data=[
                    {
                        "content": c.content,
                        "citation": c.citation.model_dump(),
                        "score": c.score,
                    }
                    for c in relevant
                ],
                source="rag_search",
                metadata={"query": query, "count": len(relevant), "retrieved": len(chunks)},
            )
        except Exception as exc:
            self.logger.exception("rag_search failed for query=%r", query)
            return ToolResult(success=False, source="rag_search", error=str(exc))
