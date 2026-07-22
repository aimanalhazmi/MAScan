"""rag_search — lets an agent query MAScan's internal RAG index.."""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mascan.contracts.retrieval import RetrievalQuery
from mascan.contracts.tools import ToolResult
from mascan.core.settings import get_settings
from mascan.rag.retriever import get_retriever
from mascan.rag.store import run_sync
from mascan.tools.base import BaseTool


class RagSearchInput(BaseModel):
    query: str = Field(..., description="Question to search the internal knowledge base for.")
    k: int = Field(5, description="Requested passages; values are clamped to 1–10.")


class RagSearchTool(BaseTool):
    name = "rag_search"
    description = (
        "Search MAScan's internal knowledge base (ingested filings, reports, and "
        "articles) and return relevant passages, each with its source citation."
    )
    input_schema: ClassVar[type[BaseModel] | None] = RagSearchInput
    MAX_RESULTS: ClassVar[int] = 10
    MAX_CONTENT_CHARS: ClassVar[int] = 1_000

    def run(self, query: str, k: int = 5, **_: Any) -> ToolResult[list[dict[str, Any]]]:
        try:
            bounded_k = max(1, min(k, self.MAX_RESULTS))
            chunks = run_sync(
                get_retriever().retrieve(RetrievalQuery(query=query, k=bounded_k))
            )
            floor = get_settings().rag_min_score
            matching = [c for c in chunks if c.score >= floor]
            relevant = matching[: self.MAX_RESULTS]
            text_truncated = any(
                isinstance(c.content, str) and len(c.content) > self.MAX_CONTENT_CHARS
                for c in relevant
            )
            return ToolResult(
                success=True,
                data=[
                    {
                        "content": self.truncate_text(c.content, self.MAX_CONTENT_CHARS),
                        "citation": c.citation.model_dump(),
                        "score": c.score,
                    }
                    for c in relevant
                ],
                source="rag_search",
                metadata={
                    "query": query,
                    "count": len(relevant),
                    "retrieved": len(chunks),
                    "limit_applied": (
                        k != bounded_k
                        or len(matching) > self.MAX_RESULTS
                        or text_truncated
                    ),
                },
            )
        except Exception as exc:
            self.logger.exception("rag_search failed for query=%r", query)
            return ToolResult(success=False, source="rag_search", error=str(exc))
