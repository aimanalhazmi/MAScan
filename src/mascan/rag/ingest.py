"""Ingestion: chunk text, embed, upsert into the PGVector store."""

from pathlib import Path

from mascan.contracts.retrieval import Chunk, Citation
from mascan.rag.chunking import chunk_text
from mascan.rag.store import (
    chunk_id,
    chunk_to_document,
    content_exists,
    content_hash,
    get_vector_store,
)


async def ingest_chunks(chunks: list[Chunk], *, key: str) -> int:
    """Embed and store chunks under a content key. Returns the number stored.

    Chunks get deterministic ids (key + position) so re-ingesting the same
    content upserts rather than duplicating.
    """
    if not chunks:
        return 0
    store = get_vector_store()
    ids = [chunk_id(key, i) for i, _ in enumerate(chunks)]
    await store.aadd_documents([chunk_to_document(c) for c in chunks], ids=ids)
    return len(chunks)


async def ingest_text(text: str, *, source: str, citation: Citation) -> int:
    """Chunk plain text and store it. Returns chunks stored (0 if already present)."""
    key = content_hash(text)
    if await content_exists(key):
        return 0
    return await ingest_chunks(chunk_text(text, source=source, citation=citation), key=key)


async def ingest_file(path: str | Path, *, document: str, source: str = "upload") -> int:
    """Ingest an uploaded file by type: .pdf via the PDF parser, .md/.txt as text.

    Returns chunks stored, or 0 if this content is already in the store.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        key = content_hash(p.read_bytes())
        if await content_exists(key):
            return 0
        from mascan.rag.parsing import parse_pdf

        return await ingest_chunks(parse_pdf(p, document=document, source=source), key=key)
    if suffix in {".md", ".txt"}:
        return await ingest_text(
            p.read_text(encoding="utf-8"),
            source=source,
            citation=Citation(document=document),
        )
    raise ValueError(f"Unsupported file type: {suffix!r} (expected .pdf, .md, .txt)")
