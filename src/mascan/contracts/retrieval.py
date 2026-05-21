"""Shapes used by the retrieval (RAG) subsystem."""

from typing import Any

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """A request for retrieval. Kept simple; extend later."""

    query: str
    k: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A single piece of retrieved content."""
    content: str
    source: str
    score: float = Field(0.0, description="Relevance score from the retriever.")
    metadata: dict[str, Any] = Field(default_factory=dict)