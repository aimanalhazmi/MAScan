"""Extract plain text from a PDF. Returns '' for image-only PDFs."""

from pathlib import Path

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - exercised when pypdf is absent.
    PdfReader = None  # type: ignore[assignment,misc]  # conditional-import fallback


def extract_pdf_text(path: str | Path) -> str:
    if PdfReader is None:
        return _extract_pdf_text_with_pymupdf(path)
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts).strip()


def _extract_pdf_text_with_pymupdf(path: str | Path) -> str:
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PDF text extraction requires either pypdf or PyMuPDF (fitz)."
        ) from exc

    parts: list[str] = []
    with fitz.open(str(path)) as document:
        for page in document:
            text = page.get_text("text") or ""
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts).strip()
