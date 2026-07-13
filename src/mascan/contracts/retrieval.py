"""Shapes used by the retrieval (RAG) subsystem."""

from typing import Any

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """A request for retrieval. Kept simple; extend later."""

    query: str
    k: int = 5


class Citation(BaseModel):
    """A single citation: where a piece of information came from."""
    document: str # filename or source id — always present
    page: int | None = None # PDF page; None for plain text
    block: str | None = None # block id on the page; None for plain text


class Chunk(BaseModel):
    """A unit of ingested content, ready to embed and store."""
    content: str
    source: str  # where the data entered from (upload, news_api, web_search...)
    citation: Citation
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(Chunk):
    """A Chunk returned by retrieval, with a relevance score."""
    score: float = Field(0.0, description="Relevance score from the retriever.")


class StoredDocument(BaseModel):
    """A document that is present in the vector store, as shown in the library."""
    document: str
    source: str
    chunks: int


class RagAnswer(BaseModel):
    """A generated answer grounded in retrieved chunks, with structured citations."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)