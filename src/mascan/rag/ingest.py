"""Ingestion: chunk text, embed, upsert into the PGVector store."""

from pathlib import Path

from mascan.contracts.retrieval import Chunk, Citation
from mascan.rag.chunking import chunk_text
from mascan.rag.store import chunk_to_document, get_vector_store


async def ingest_chunks(chunks: list[Chunk]) -> int:
    """Embed and store chunks. Returns the number stored."""
    if not chunks:
        return 0
    store = get_vector_store()
    await store.aadd_documents([chunk_to_document(c) for c in chunks])
    return len(chunks)


async def ingest_text(text: str, *, source: str, citation: Citation) -> int:
    """Chunk plain text and store it. Returns chunks stored."""
    return await ingest_chunks(chunk_text(text, source=source, citation=citation))


async def ingest_file(path: str | Path, *, document: str, source: str = "upload") -> int:
    """Ingest an uploaded file by type: .pdf via the PDF parser, .md/.txt as text."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        from mascan.rag.parsing import parse_pdf

        return await ingest_chunks(parse_pdf(p, document=document, source=source))
    if suffix in {".md", ".txt"}:
        return await ingest_text(
            p.read_text(encoding="utf-8"),
            source=source,
            citation=Citation(document=document),
        )
    raise ValueError(f"Unsupported file type: {suffix!r} (expected .pdf, .md, .txt)")
