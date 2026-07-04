"""Answer generation: retrieve, build a grounded prompt, and answer with citations.

The model answers only from the numbered documents and cites them as [n]. Each
cited document's `Citation` is returned alongside the text.
"""

import base64
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mascan.contracts.retrieval import Citation, RagAnswer, RetrievalQuery, RetrievedChunk
from mascan.core.llm import get_chat_model, get_vision_model
from mascan.core.logging import get_logger
from mascan.rag.retriever import get_retriever

logger = get_logger("rag.answer")

# Cap attached images so figure-heavy chunks can't blow up the request size.
MAX_IMAGES = 4

SYSTEM_PROMPT = """\
You answer questions using only the numbered documents provided.
- Use only facts found in the documents. Do not invent or use outside knowledge.
- Some documents include figures attached as images; use them when relevant.
- Cite the documents you rely on inline as [1], [2], matching their numbers.
- If the documents do not contain the answer, say so plainly. Do not guess.
- Be concise. No filler, no restating the question.
"""


def collect_image_blocks(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Build de-duplicated image_url content blocks from chunks' figure images."""
    seen: set[str] = set()
    blocks: list[dict[str, Any]] = []
    for chunk in chunks:
        for path in chunk.metadata.get("images", []):
            if path in seen:
                continue
            seen.add(path)
            p = Path(path)
            if not p.exists():
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            blocks.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
            if len(blocks) >= MAX_IMAGES:
                return blocks
    return blocks


CITE_RE = re.compile(r"\[(\d+)\]")


def select_cited(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Keep only the citations the answer references, renumbered [1..m].

    Returns the renumbered text and matching citations, so [n] always maps to
    citations[n-1]. No markers means no citations.
    """
    order: list[int] = []
    for match in CITE_RE.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(chunks) and n not in order:
            order.append(n)
    if not order:
        return answer, []

    remap = {old: new for new, old in enumerate(order, start=1)}
    new_answer = CITE_RE.sub(
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0),
        answer,
    )
    citations = [chunks[old - 1].citation for old in order]
    return new_answer, citations


def citation_tag(c: Citation) -> str:
    """Human-readable provenance for the context header, e.g. '10k.pdf p.4 (table-2)'."""
    parts = [c.document]
    if c.page is not None:
        parts.append(f"p.{c.page}")
    if c.block:
        parts.append(f"({c.block})")
    return " ".join(parts)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the chunks so the model can cite them as [n]."""
    return "\n\n".join(
        f"[{i}] {citation_tag(c.citation)}\n{c.content}" for i, c in enumerate(chunks, start=1)
    )


async def answer_question(query: RetrievalQuery) -> RagAnswer:
    """Retrieve, generate a grounded answer, and return it with the chunks' citations."""
    chunks = await get_retriever().retrieve(query)
    if not chunks:
        return RagAnswer(answer="I don't have enough indexed information to answer that.")

    prompt = f"Documents:\n{format_context(chunks)}\n\nQuestion: {query.query}\n\nAnswer:"

    # Multimodal only when a retrieved chunk actually carries a figure image.
    image_blocks = collect_image_blocks(chunks)
    if image_blocks:
        llm = get_vision_model(temperature=0.2)
        content: str | list[dict[str, Any]] = [{"type": "text", "text": prompt}, *image_blocks]
    else:
        llm = get_chat_model(temperature=0.2)  # default text model
        content = prompt

    response = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)]
    )
    # Keep only the citations the model used; the full set stays in `chunks`.
    answer, citations = select_cited(str(response.content), chunks)
    return RagAnswer(answer=answer, citations=citations, chunks=chunks)
