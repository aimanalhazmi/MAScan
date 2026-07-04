"""Text chunking for ingestion.

Recursive character splitting to keep sentences and paragraphs together, then
wraps each piece in a `Chunk` carrying its citation.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mascan.contracts.retrieval import Chunk, Citation
from mascan.core.settings import get_settings

settings = get_settings()
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_text(text: str, *, source: str, citation: Citation) -> list[Chunk]:
    """Split text into overlapping chunks, each carrying its citation.

    The citation applies to the whole text (e.g. one news article). For PDFs,
    call this per page with that page's citation.
    """
    return [
        Chunk(content=piece, source=source, citation=citation)
        for piece in splitter.split_text(text)
    ]
