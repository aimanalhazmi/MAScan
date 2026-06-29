"""Shapes used by the retrieval (RAG) subsystem."""

from typing import Any

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """A request for retrieval. Kept simple; extend later."""

    query: str
    k: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A single citation: where a piece of information came from."""
    document: str # filename or source id — always present
    page: int | None = None # PDF page; None for plain text
    block: str | None = None # block id on the page; None for plain text


class RetrievedChunk(BaseModel):
    """A single piece of retrieved content."""
    content: str
    source: str # where the data entered from (upload, news_api, web_search...)
    citation: Citation
    score: float = Field(0.0, description="Relevance score from the retriever.")
    metadata: dict[str, Any] = Field(default_factory=dict)