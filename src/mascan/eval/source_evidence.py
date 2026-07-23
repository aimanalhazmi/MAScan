"""Validate gold-standard source anchors against extracted PDF text."""

import re
from pathlib import Path

from pydantic import BaseModel, Field

from mascan.eval.gold_standard import GoldStandardDataset
from mascan.eval.pdf import extract_pdf_text

DEFAULT_ANCHOR_MATCH_THRESHOLD = 0.45
SOURCE_EVIDENCE_CSV_FIELDS = [
    "case_id",
    "source_pdf",
    "anchor",
    "matched",
    "score",
    "direct_phrase_match",
    "matched_terms",
    "missing_terms",
    "excerpt",
    "error",
]

_STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "and",
    "are",
    "between",
    "both",
    "can",
    "connect",
    "connects",
    "describe",
    "describes",
    "detail",
    "details",
    "for",
    "from",
    "has",
    "identify",
    "identifies",
    "including",
    "into",
    "section",
    "sections",
    "support",
    "supports",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "with",
}


class SourceAnchorEvidence(BaseModel):
    case_id: str
    source_pdf: str
    anchor: str
    matched: bool
    score: float = Field(..., ge=0.0, le=1.0)
    direct_phrase_match: bool = False
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    excerpt: str | None = None
    error: str | None = None


class SourceEvidenceReport(BaseModel):
    case_count: int
    anchor_count: int
    matched_anchor_count: int
    unmatched_anchor_count: int
    extract_error_count: int
    match_threshold: float
    average_score: float | None = None
    evidence: list[SourceAnchorEvidence] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.unmatched_anchor_count == 0 and self.extract_error_count == 0


def validate_source_anchor_evidence(
    dataset: GoldStandardDataset,
    *,
    base_dir: str | Path = ".",
    match_threshold: float = DEFAULT_ANCHOR_MATCH_THRESHOLD,
) -> SourceEvidenceReport:
    """Check every validation-note source anchor against its source PDF text."""
    base = Path(base_dir)
    text_cache: dict[str, str] = {}
    extract_errors: dict[str, str] = {}
    evidence: list[SourceAnchorEvidence] = []

    for case in dataset.cases:
        pdf_path = _resolve(base, case.source_pdf)
        pdf_key = str(pdf_path)
        if pdf_key not in text_cache and pdf_key not in extract_errors:
            try:
                text = extract_pdf_text(pdf_path)
                if not text.strip():
                    extract_errors[pdf_key] = "No extractable text found in PDF."
                else:
                    text_cache[pdf_key] = text
            except Exception as exc:
                extract_errors[pdf_key] = str(exc)

        for anchor in case.validation_notes.source_anchors:
            if pdf_key in extract_errors:
                evidence.append(
                    SourceAnchorEvidence(
                        case_id=case.case_id,
                        source_pdf=case.source_pdf,
                        anchor=anchor,
                        matched=False,
                        score=0.0,
                        error=extract_errors[pdf_key],
                    )
                )
                continue
            evidence.append(
                _score_anchor(
                    case_id=case.case_id,
                    source_pdf=case.source_pdf,
                    anchor=anchor,
                    pdf_text=text_cache[pdf_key],
                    match_threshold=match_threshold,
                )
            )

    matched = sum(1 for item in evidence if item.matched)
    scores = [item.score for item in evidence]
    return SourceEvidenceReport(
        case_count=len(dataset.cases),
        anchor_count=len(evidence),
        matched_anchor_count=matched,
        unmatched_anchor_count=len(evidence) - matched,
        extract_error_count=sum(1 for item in evidence if item.error),
        match_threshold=match_threshold,
        average_score=round(sum(scores) / len(scores), 6) if scores else None,
        evidence=evidence,
    )


def render_source_evidence_markdown(report: SourceEvidenceReport) -> str:
    lines = [
        "# Gold-Standard Source Anchor Evidence",
        "",
        f"- Cases: {report.case_count}",
        f"- Anchors: {report.anchor_count}",
        f"- Matched anchors: {report.matched_anchor_count}",
        f"- Unmatched anchors: {report.unmatched_anchor_count}",
        f"- Extraction errors: {report.extract_error_count}",
        f"- Match threshold: {report.match_threshold:.2f}",
        f"- Average score: {_fmt(report.average_score)}",
        "",
        "| Case | Matched | Score | Anchor | Matched Terms | Missing Terms | Excerpt/Error |",
        "|---|---|---:|---|---|---|---|",
    ]
    for item in report.evidence:
        excerpt_or_error = item.error or item.excerpt or "-"
        lines.append(
            "| "
            f"{item.case_id} | "
            f"{_yes_no(item.matched)} | "
            f"{item.score:.4f} | "
            f"{_cell(item.anchor)} | "
            f"{_cell(', '.join(item.matched_terms) or '-')} | "
            f"{_cell(', '.join(item.missing_terms) or '-')} | "
            f"{_cell(excerpt_or_error)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def source_evidence_csv_rows(
    report: SourceEvidenceReport,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in report.evidence:
        rows.append(
            {
                "case_id": item.case_id,
                "source_pdf": item.source_pdf,
                "anchor": item.anchor,
                "matched": str(item.matched).lower(),
                "score": f"{item.score:.6f}",
                "direct_phrase_match": str(item.direct_phrase_match).lower(),
                "matched_terms": "; ".join(item.matched_terms),
                "missing_terms": "; ".join(item.missing_terms),
                "excerpt": item.excerpt or "",
                "error": item.error or "",
            }
        )
    return rows


def _score_anchor(
    *,
    case_id: str,
    source_pdf: str,
    anchor: str,
    pdf_text: str,
    match_threshold: float,
) -> SourceAnchorEvidence:
    anchor_terms = _content_terms(anchor)
    pdf_keys = _term_key_set(_content_terms(pdf_text))
    matched_terms = sorted(term for term in anchor_terms if _term_keys(term) & pdf_keys)
    missing_terms = sorted(set(anchor_terms) - set(matched_terms))
    score = len(matched_terms) / len(anchor_terms) if anchor_terms else 0.0
    direct_phrase_match = _normalized_phrase(anchor) in _normalized_phrase(pdf_text)
    matched = direct_phrase_match or score >= match_threshold
    return SourceAnchorEvidence(
        case_id=case_id,
        source_pdf=source_pdf,
        anchor=anchor,
        matched=matched,
        score=round(score, 6),
        direct_phrase_match=direct_phrase_match,
        matched_terms=matched_terms,
        missing_terms=missing_terms,
        excerpt=_best_excerpt(pdf_text, set(anchor_terms)),
    )


def _content_terms(text: str) -> list[str]:
    normalized = re.sub(r"\bR\s*&\s*D\b", "research development", text)
    normalized = normalized.replace("-", " ")
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9']{2,}", normalized)
    terms: set[str] = set()
    for raw_term in raw_terms:
        term = raw_term.lower()
        if term in _STOPWORDS:
            continue
        if len(term) >= 4 or (len(term) >= 3 and raw_term.isupper()):
            terms.add(term)
    return sorted(terms)


def _best_excerpt(text: str, anchor_terms: set[str], *, max_length: int = 420) -> str | None:
    if not anchor_terms:
        return None
    anchor_keys = _term_key_set(anchor_terms)
    chunks = _chunks(text)
    if not chunks:
        return None
    best = max(
        chunks,
        key=lambda chunk: len(_term_key_set(_content_terms(chunk)) & anchor_keys),
    )
    compact = re.sub(r"\s+", " ", best).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def _chunks(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= 1200:
            chunks.append(paragraph)
            continue
        for start in range(0, len(paragraph), 900):
            chunk = paragraph[start : start + 1200].strip()
            if chunk:
                chunks.append(chunk)
    return chunks


def _term_key_set(terms: set[str] | list[str]) -> set[str]:
    keys: set[str] = set()
    for term in terms:
        keys.update(_term_keys(term))
    return keys


def _term_keys(term: str) -> set[str]:
    keys = {term}
    if len(term) >= 7:
        keys.add(term[:7])
    return keys


def _normalized_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _resolve(base: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
