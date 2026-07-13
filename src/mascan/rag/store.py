"""PGVector store backed by OpenAI embeddings."""

import asyncio
import hashlib
import threading
from collections.abc import Coroutine
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from pydantic import SecretStr
from sqlalchemy import text

from mascan.contracts.retrieval import Chunk, Citation, RetrievedChunk, StoredDocument
from mascan.core.exceptions import ConfigError
from mascan.core.settings import get_settings

rag_loop: asyncio.AbstractEventLoop | None = None
loop_lock = threading.Lock()
store: PGVector | None = None


def get_rag_loop() -> asyncio.AbstractEventLoop:
    """The one event loop all RAG work runs on.

    The store's connection pool and the embedding client bind to the loop that first
    uses them, and fail with a connection error when reached from another loop. So
    everything goes through this loop: the API's async endpoints via `on_rag_loop`,
    the sync callers (agents, planner, CLI) via `run_sync`. Callers with no loop of
    their own — the orchestrator script — need it to exist anyway.
    """
    global rag_loop
    with loop_lock:
        if rag_loop is None:
            rag_loop = asyncio.new_event_loop()
            threading.Thread(target=rag_loop.run_forever, name="rag-loop", daemon=True).start()
        return rag_loop


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run RAG from sync code (agents, planner, CLI) and wait for it."""
    return asyncio.run_coroutine_threadsafe(coro, get_rag_loop()).result()


async def on_rag_loop[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run RAG from async code (FastAPI) without blocking the caller's loop."""
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, get_rag_loop()))


def content_hash(content: str | bytes) -> str:
    """Stable key for a document's content, so identical content dedupes
    regardless of filename."""
    data = content.encode() if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def chunk_id(key: str, index: int) -> str:
    """Deterministic chunk id from content key + position, so re-ingesting the
    same content upserts instead of creating duplicate rows."""
    return hashlib.sha256(f"{key}\x00{index}".encode()).hexdigest()


async def content_exists(key: str) -> bool:
    """True if content with this key is already stored (checks its first chunk)."""
    got = await get_vector_store().aget_by_ids([chunk_id(key, 0)])
    return bool(got)


async def list_documents(source: str | None = None) -> list[StoredDocument]:
    """List the documents in the collection, grouped by citation document.

    Optionally restricted to one source (e.g. "upload"), so the document library
    shows what the user uploaded and not what agents ingested along the way.
    """
    sql = text("""
        SELECT e.cmetadata #>> '{citation,document}' AS document,
               e.cmetadata ->> 'source'              AS source,
               count(*)                              AS chunks
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :collection
          AND (CAST(:source AS text) IS NULL OR e.cmetadata ->> 'source' = :source)
        GROUP BY 1, 2
        ORDER BY 1
    """)
    params = {"collection": get_settings().rag_collection, "source": source}

    async with get_vector_store().session_maker() as session:
        rows = (await session.execute(sql, params)).all()

    return [
        StoredDocument(document=name, source=src or "", chunks=count)
        for name, src, count in rows
        if name
    ]


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


def document_to_retrieved_chunk(doc: Document, distance: float) -> RetrievedChunk:
    """Inverse of chunk_to_document, attaching the retriever's relevance score.

    PGVector returns a cosine *distance* (lower = closer). We expose it as a
    similarity in [0, 1] where higher = more relevant, so callers (and `merge`,
    which keeps the highest score) sort the intuitive way.
    """
    meta = dict(doc.metadata)
    source = meta.pop("source", "")
    citation = Citation(**meta.pop("citation", {"document": source}))
    return RetrievedChunk(
        content=doc.page_content,
        source=source,
        citation=citation,
        metadata=meta,
        score=1.0 - distance,
    )


def get_vector_store() -> PGVector:
    """
    Retrieve or initialize a global instance of the PGVector store.

    This function checks if the global `store` variable has already been initialized.
    If not, it creates a new PGVector instance using the settings provided by the
    application configuration.

    Raises:
        ConfigError: If the `DATABASE_URL` is not set in the application settings.

    :return: An initialized instance of PGVector.
    :rtype: PGVector
    """
    global store
    s = get_settings()
    if not s.database_url:
        raise ConfigError("DATABASE_URL not set; cannot build the PGVector store.")

    if store is None:
        embeddings = OpenAIEmbeddings(model=s.embedding_model, api_key=SecretStr(s.openai_api_key))
        store = PGVector(
            embeddings=embeddings,
            connection=s.database_url,
            collection_name=s.rag_collection,
            embedding_length=s.embedding_dim,
            use_jsonb=True,
            async_mode=True,
        )
    return store
