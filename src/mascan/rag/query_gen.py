"""Query expansion to widen recall before dense search.

Two LLM steps:
- `decompose_query`: split a multi-hop question into standalone sub-questions.
  Single-step questions pass through unchanged.
- `generate_hyde_documents`: write a few short, distinct hypothetical answer
  passages (Multi-HyDE), so retrieval matches answer-like text, not just the
  terse question.
"""

from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger

logger = get_logger("rag.query_gen")

# Small fan-out caps: every extra query means more dense searches. Raise if
# recall needs it.
MAX_SUBQUERIES = 3
NUM_HYDE_DOCS = 3

DECOMPOSE_PROMPT = """\
Break the question into the minimal set of standalone sub-questions needed to \
answer it. If it is already a single-step question, return it unchanged. \
Return one sub-question per line, no numbering, no extra text.

Question: {question}
"""

HYDE_PROMPT = """\
Write {n} short, NON-equivalent hypothetical answer passages for the question \
below. Each passage should read like an excerpt from a document that answers \
the question, and each must cover a DISTINCT aspect — do not paraphrase the \
others. Two or three sentences each. Separate passages with a blank line.

Question: {question}
"""


def parse_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


async def decompose_query(question: str) -> list[str]:
    """Return standalone sub-questions; falls back to [question] on any failure."""
    llm = get_chat_model(temperature=0.0)
    resp = await llm.ainvoke(DECOMPOSE_PROMPT.format(question=question))
    subs = parse_lines(str(resp.content))[:MAX_SUBQUERIES]
    return subs or [question]


async def generate_hyde_documents(question: str) -> list[str]:
    """Return multiple non-equivalent hypothetical passages (Multi-HyDE)."""
    llm = get_chat_model(temperature=0.7)  # some diversity across passages
    resp = await llm.ainvoke(HYDE_PROMPT.format(question=question, n=NUM_HYDE_DOCS))
    # fall back to the raw question if nothing parses.
    blocks = [b.strip() for b in str(resp.content).split("\n\n") if b.strip()]
    return blocks[:NUM_HYDE_DOCS] or [question]
