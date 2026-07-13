"""PGVector store backed by OpenAI embeddings."""

import asyncio
import hashlib

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from pydantic import SecretStr
from sqlalchemy import text

from mascan.contracts.retrieval import Chunk, Citation, RetrievedChunk, StoredDocument
from mascan.core.exceptions import ConfigError
from mascan.core.settings import get_settings

stores: dict[int, PGVector] = {}


DOCUMENTS_SQL = text("""
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
    store = get_vector_store()
    params = {"collection": get_settings().rag_collection, "source": source}

    async with store.session_maker() as session:
        rows = (await session.execute(DOCUMENTS_SQL, params)).all()

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
    """Return the async PGVector store for the running event loop.

    The store's database engine and its embedding client bind to the loop that
    first uses them, so one store per loop: the API serves requests on FastAPI's
    loop, while agents and the planner reach the same data from their own loop.
    Sharing a single store between the two raises a connection error.

    Raises ConfigError if no DATABASE_URL is configured.
    """
    s = get_settings()
    if not s.database_url:
        raise ConfigError("DATABASE_URL not set; cannot build the PGVector store.")

    try:
        key = id(asyncio.get_running_loop())
    except RuntimeError:
        key = 0

    store = stores.get(key)
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
        stores[key] = store
    return store
