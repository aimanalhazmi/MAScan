"""PGVector store backed by OpenAI embeddings."""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from pydantic import SecretStr

from mascan.contracts.retrieval import Chunk, Citation, RetrievedChunk
from mascan.core.exceptions import ConfigError
from mascan.core.settings import get_settings


def chunk_to_document(chunk: Chunk) -> Document:
    """Map a Chunk to a langchain Document; citation and source go into metadata."""
    return Document(
        page_content=chunk.content,
        metadata={
            "source": chunk.source,
            "citation": chunk.citation.model_dump(),
            **chunk.metadata,
        },
    )


def document_to_retrieved_chunk(doc: Document, score: float) -> RetrievedChunk:
    """Inverse of chunk_to_document, attaching the retriever's relevance score."""
    meta = dict(doc.metadata)
    source = meta.pop("source", "")
    citation = Citation(**meta.pop("citation", {"document": source}))
    return RetrievedChunk(
        content=doc.page_content,
        source=source,
        citation=citation,
        metadata=meta,
        score=score,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    """Return a cached async PGVector store.

    Raises ConfigError if no DATABASE_URL is configured.
    """
    s = get_settings()
    if not s.database_url:
        raise ConfigError("DATABASE_URL not set; cannot build the PGVector store.")
    embeddings = OpenAIEmbeddings(model=s.embedding_model, api_key=SecretStr(s.openai_api_key))
    return PGVector(
        embeddings=embeddings,
        connection=s.database_url,
        collection_name=s.rag_collection,
        embedding_length=s.embedding_dim,
        use_jsonb=True,
        async_mode=True,
    )
