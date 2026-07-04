"""PDF parsing for uploads.

Uses PyMuPDF (fitz) to pull per-page text and embedded figure images. Text is
chunked with a page-level citation, and each chunk records its page's image
paths in metadata["images"] so the answer step can pass them to a vision model.
"""

import base64
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage

from mascan.contracts.retrieval import Chunk, Citation
from mascan.core.llm import get_vision_model
from mascan.core.logging import get_logger
from mascan.core.settings import get_settings
from mascan.rag.chunking import chunk_text

logger = get_logger("rag.parsing")

CAPTION_PROMPT = (
    "Describe this figure for retrieval: what it shows, chart/table type, axes, "
    "labels, and any key numbers or trends. Be concise and factual."
)


def caption_figure(image_path: str) -> str:
    """Describe a figure in text so it can be retrieved by its content.

    One vision call per figure. Returns an empty string on any failure.
    """
    try:
        data = base64.b64encode(Path(image_path).read_bytes()).decode()
        msg = HumanMessage(
            content=[
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
            ]
        )
        return str(get_vision_model(temperature=0.0).invoke([msg]).content).strip()
    except Exception:
        logger.warning("figure captioning failed for %s", image_path, exc_info=True)
        return ""


def extract_page_images(page: Any, page_no: int, out_dir: Path) -> list[str]:
    """Save the page's embedded raster images to disk; return their paths."""
    paths: list[str] = []
    for idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(page.parent, xref)
        if pix.n - pix.alpha >= 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        path = out_dir / f"p{page_no}-img{idx}.png"
        pix.save(str(path))
        paths.append(str(path))
    return paths


def parse_pdf(path: str | Path, *, document: str, source: str = "upload") -> list[Chunk]:
    """Parse a PDF into chunks with page-level citations and figure images.

    Each page's text is chunked; every chunk from a page carries that page's
    extracted image paths in metadata["images"].
    """
    out_dir = Path(get_settings().rag_image_dir) / document
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text")
            images = extract_page_images(page, page_no, out_dir)
            if not text.strip() and not images:
                continue

            page_chunks = chunk_text(
                text, source=source, citation=Citation(document=document, page=page_no)
            )
            # Attach the page's figures to each of its chunks.
            for c in page_chunks:
                if images:
                    c.metadata["images"] = images
            chunks.extend(page_chunks)

            # A caption chunk per figure, so figures are searchable on their own.
            for img in images:
                caption = caption_figure(img)
                content = caption or f"[Figure on page {page_no} of {document}]"
                chunks.append(
                    Chunk(
                        content=content,
                        source=source,
                        citation=Citation(document=document, page=page_no, block="figure"),
                        metadata={"images": [img]},
                    )
                )
    return chunks
