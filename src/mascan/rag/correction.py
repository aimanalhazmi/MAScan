"""Self-correction for retrieval (CRAG).

Grade whether the retrieved chunks actually answer the question; if not, rewrite
the query so the caller can retry..
"""

from mascan.contracts.retrieval import RetrievedChunk
from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger

logger = get_logger("rag.correction")

SNIPPET_CHARS = 500

GRADE_PROMPT = """\
Do the passages below contain enough information to answer the question?
Answer with a single word: YES or NO.

Question: {question}

{documents}
"""

REWRITE_PROMPT = """\
The following search query did not retrieve useful results. Rewrite it into a \
clearer, more specific query that is likely to match relevant documents. \
Return only the rewritten query, nothing else.

Query: {question}
"""


async def grade_relevance(question: str, chunks: list[RetrievedChunk]) -> bool:
    """True if the chunks plausibly answer the question (LLM yes/no grade)."""
    if not chunks:
        return False
    documents = "\n\n".join(
        f"[{i}] {c.content[:SNIPPET_CHARS]}" for i, c in enumerate(chunks, start=1)
    )
    llm = get_chat_model(temperature=0.0)
    resp = await llm.ainvoke(GRADE_PROMPT.format(question=question, documents=documents))
    return "yes" in str(resp.content).strip().lower()[:5]


async def rewrite_query(question: str) -> str:
    """Rewrite a weak query into a more retrievable one; falls back to the input."""
    llm = get_chat_model(temperature=0.0)
    resp = await llm.ainvoke(REWRITE_PROMPT.format(question=question))
    return str(resp.content).strip() or question
