"""Deterministic claim-citation extraction for final Markdown reports."""

import re
from dataclasses import dataclass

from markdown_it import MarkdownIt

from mascan.agents.sources import canonical_source_url

SUMMARY_HEADING_RE = re.compile(r"(?im)^##\s+Summary\s*$")
END_HEADING_RE = re.compile(r"(?im)^##\s+(?:Sources|Fact Check)\s*$")
MARKDOWN_CITATION_RE = re.compile(r"\[(\d+)\]\((https?://[^)\s]+)\)")
MARKDOWN_CITATION_NUMBER_RE = re.compile(r"\[(\d+)\]\(")
BARE_CITATION_RE = re.compile(r"\[(\d+)\](?!\()")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[\"'(*_]*[A-Z0-9]))")
MARKDOWN_DECORATION_RE = re.compile(r"(?:\*\*|__|~~|`)")


@dataclass(frozen=True)
class CitationRef:
    number: int
    url: str
    canonical_url: str


@dataclass(frozen=True)
class Attribution:
    claim: str
    passage: str
    citations: tuple[CitationRef, ...]


@dataclass(frozen=True)
class AttributionDocument:
    body: str
    citations: tuple[CitationRef, ...]
    attributions: tuple[Attribution, ...]
    uncited_claims: tuple[str, ...]


def extract_summary_body(markdown: str) -> str:
    """Return only the final report body between Summary and Sources/Fact Check."""
    text = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    summary = SUMMARY_HEADING_RE.search(text)
    start = summary.end() if summary else 0
    end_match = END_HEADING_RE.search(text, pos=start)
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def parse_attribution_document(markdown: str) -> AttributionDocument:
    """Parse numbered inline citations and associate them with sentence claims.

    Markdown AST traversal restricts matching to real paragraph/list-item inline
    nodes. Fenced code and report sections outside Summary are therefore ignored.
    A citation at the end of a passage is attributed backwards to preceding
    uncited sentences in that same passage.
    """
    body = extract_summary_body(markdown)
    tokens = MarkdownIt("commonmark").parse(body)
    citations_by_url: dict[str, CitationRef] = {}
    attributions: list[Attribution] = []
    uncited: list[str] = []

    for index, token in enumerate(tokens):
        if token.type != "inline" or index == 0:
            continue
        opener = tokens[index - 1]
        if opener.type != "paragraph_open":
            continue

        passage = token.content.strip()
        if not passage:
            continue
        ast_refs = _citation_refs_from_inline_token(token)
        ast_refs.extend(
            CitationRef(
                number=int(number),
                url=f"uploaded-document:{number}",
                canonical_url=f"uploaded-document:{number}",
            )
            for number in BARE_CITATION_RE.findall(passage)
        )
        ast_refs = _dedupe_refs(ast_refs)
        if not ast_refs:
            uncited.extend(_plain_claims(passage))
            continue
        for ref in ast_refs:
            if ref.canonical_url and ref.canonical_url not in citations_by_url:
                citations_by_url[ref.canonical_url] = ref

        refs_by_number: dict[int, list[CitationRef]] = {}
        for ref in ast_refs:
            refs_by_number.setdefault(ref.number, []).append(ref)
        sentences = _split_sentences(passage)
        sentence_refs: list[list[CitationRef]] = []
        for sentence in sentences:
            linked_numbers = [int(number) for number in MARKDOWN_CITATION_NUMBER_RE.findall(sentence)]
            bare_numbers = [int(number) for number in BARE_CITATION_RE.findall(sentence)]
            refs = [
                ref
                for number in [*linked_numbers, *bare_numbers]
                for ref in refs_by_number.get(number, [])
            ]
            sentence_refs.append(_dedupe_refs(refs))

        if sentence_refs and sentence_refs[-1]:
            trailing_refs = sentence_refs[-1]
            sentence_refs = [refs or trailing_refs for refs in sentence_refs]

        for sentence, refs in zip(sentences, sentence_refs, strict=True):
            claim = _clean_claim(sentence)
            if not claim:
                continue
            if not refs:
                uncited.append(claim)
                continue
            attributions.append(
                Attribution(claim=claim, passage=passage, citations=tuple(refs))
            )

    return AttributionDocument(
        body=body,
        citations=tuple(citations_by_url.values()),
        attributions=tuple(attributions),
        uncited_claims=tuple(dict.fromkeys(uncited)),
    )


def _citation_refs_from_inline_token(token: object) -> list[CitationRef]:
    children = getattr(token, "children", None) or []
    refs: list[CitationRef] = []
    index = 0
    while index < len(children):
        child = children[index]
        if child.type != "link_open":
            index += 1
            continue
        url = child.attrGet("href") or ""
        label_parts: list[str] = []
        index += 1
        while index < len(children) and children[index].type != "link_close":
            label_parts.append(children[index].content or "")
            index += 1
        label = "".join(label_parts).strip()
        if label.isdigit() and url.startswith(("http://", "https://")):
            refs.append(
                CitationRef(
                    number=int(label),
                    url=url,
                    canonical_url=canonical_source_url(url),
                )
            )
        index += 1
    return _dedupe_refs(refs)


def _split_sentences(passage: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY_RE.split(passage) if part.strip()]


def _plain_claims(passage: str) -> list[str]:
    return [claim for sentence in _split_sentences(passage) if (claim := _clean_claim(sentence))]


def _clean_claim(text: str) -> str:
    without_citations = MARKDOWN_CITATION_RE.sub("", text)
    without_citations = BARE_CITATION_RE.sub("", without_citations)
    without_markup = MARKDOWN_DECORATION_RE.sub("", without_citations)
    normalized = " ".join(without_markup.split()).strip(" -*_")
    return re.sub(r"\s+([,.;:!?])", r"\1", normalized)


def _dedupe_refs(refs: list[CitationRef]) -> list[CitationRef]:
    seen: set[tuple[int, str]] = set()
    result: list[CitationRef] = []
    for ref in refs:
        key = (ref.number, ref.canonical_url)
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result
