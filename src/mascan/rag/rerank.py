"""LLM reranking.

Ask the chat model to order the candidate chunks by relevance to the question,
then keep the top-k.
"""

import re

from mascan.contracts.retrieval import RetrievedChunk
from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger
from mascan.core.settings import get_settings

logger = get_logger("rag.rerank")

# Show the ranker a full chunk, not half of one — truncating below chunk_size
# hid answers living in the back half of a chunk from the LLM.
SNIPPET_CHARS = get_settings().chunk_size

PROMPT = """\
Rank the documents by how well they help answer the question. Return only the \
document numbers, most relevant first, comma-separated (e.g. 3,1,5). Include \
only documents that are actually relevant.

Question: {question}

{documents}
"""


async def rerank(question: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Reorder chunks by LLM-judged relevance and return the top_k.

    Falls back to the input order (truncated to top_k) if the LLM output cannot
    be parsed.
    """
    if len(chunks) <= 1:
        return chunks[:top_k]
    documents = "\n\n".join(
        f"[{i}] {c.content[:SNIPPET_CHARS]}" for i, c in enumerate(chunks, start=1)
    )
    llm = get_chat_model(temperature=0.0)
    resp = await llm.ainvoke(PROMPT.format(question=question, documents=documents))

    seen: set[int] = set()
    ranked: list[RetrievedChunk] = []
    for n in (int(m) for m in re.findall(r"\d+", str(resp.content))):
        if 1 <= n <= len(chunks) and n not in seen:
            seen.add(n)
            ranked.append(chunks[n - 1])
    if not ranked:  # parse failure → keep original order
        return chunks[:top_k]
    return ranked[:top_k]
