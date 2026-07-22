# RAG

Retrieval-Augmented Generation for MAScan. It ingests documents (text and
PDFs), stores them as embeddings, retrieves the relevant passages for a
question, and generates an answer grounded in those passages with source
citations.

RAG is optional. Without `DATABASE_URL` set, retrieval returns nothing
(`NullRetriever`) and the rest of MAScan runs unchanged.

## Enable and use

1. Set `DATABASE_URL` in `.env`. Docker Compose already provides a Postgres
   (pgvector) instance and sets this for the API container, so RAG is on by
   default when you run `make compose-up`.
2. Add documents (text or PDF) through the API:
   - `POST /rag/upload` — upload a file, or `POST /rag/ingest` — ingest text.
   - `GET /rag/documents` — list what is indexed.
3. Ask grounded questions with `POST /rag/answer`, or let the planner and agents
   search the index automatically (see "How the planner uses it" below).

PDF figures are read and captioned by a vision model (defaults to OpenAI
`gpt-4o`). To cut cost, point `VISION_BASE_URL`, `VISION_API_KEY`, and
`RAG_VISION_MODEL` at a self-hosted OpenAI-compatible model.

## Modules

| File | Responsibility |
|---|---|
| `ingest.py` | Entry points to ingest text or a file (`.pdf`, `.md`, `.txt`). |
| `chunking.py` | Split text into overlapping chunks (recursive character splitting). |
| `parsing.py` | PDF → per-page text + figure images; captions figures with a vision model. |
| `store.py` | PGVector store (OpenAI embeddings) and `Chunk` ↔ langchain `Document` mapping. |
| `retriever.py` | Retrieval strategies and the assembled pipeline (`get_retriever`). |
| `query_gen.py` | Query decomposition and Multi-HyDE query expansion. |
| `rerank.py` | LLM listwise reranking of candidates. |
| `correction.py` | Self-correction (CRAG): grade results, rewrite the query, retry. |
| `answer.py` | Grounded answer generation with inline `[n]` citations. |

Shared data shapes (`Chunk`, `Citation`, `RetrievedChunk`, `RagAnswer`,
`RetrievalQuery`) live in `mascan/contracts/retrieval.py`.

## How retrieval works

`get_retriever()` returns an **adaptive** pipeline:

1. **Cheap pass** — dense similarity search + LLM rerank.
2. Grade the result against the question. If it is good enough, return it.
3. **Full pass** (only when the cheap pass is weak) —
   decompose → Multi-HyDE → fan-out dense search → rerank → CRAG self-correction.

Each strategy is a small class that wraps another retriever, so the pipeline
is just composition.

## How answering works

`answer_question(query)` retrieves chunks, builds a numbered context, and asks
the LLM to answer using only those documents and cite them as `[n]`. Chunks
carrying figure images are sent to a vision model. Only the citations the
answer actually references are returned, renumbered `[1..m]`.

## Usage

```python
from mascan.contracts.retrieval import RetrievalQuery
from mascan.rag.ingest import ingest_file, ingest_text
from mascan.rag.answer import answer_question

await ingest_file("report.pdf", document="report.pdf")
result = await answer_question(RetrievalQuery(query="What are the main risks?"))
print(result.answer, result.citations)
```

## How the planner uses it

`rag_search` (`mascan/tools/common/rag_search.py`) is bound to the planner. Before
it writes the plan, the planner decides for itself whether the knowledge base holds
context about the request (the user's company, product, or market) and searches it
if so. Passages above `RAG_MIN_SCORE` go into the planning prompt, and the planner
carries what matters into each agent's `objective_context`. When it looks up nothing,
or nothing relevant comes back, planning continues unchanged.

Agents can search the index with the same tool by listing it in their `config.yaml`.

HTTP endpoints (see `mascan/app/api.py`): `POST /rag/ingest`, `/rag/upload`,
`/rag/search`, `/rag/answer`, and `GET /rag/documents` (the document library:
what is indexed, grouped by document, uploads only by default).

## Configuration

Set via environment / `mascan.core.settings`:

| Setting | Purpose |
|---|---|
| `DATABASE_URL` | Async Postgres URL. Unset → RAG disabled. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | OpenAI embedding model and vector length. |
| `RAG_COLLECTION` | PGVector collection name. |
| `RAG_VISION_MODEL` | Vision model for reading/captioning PDF figures. |
| `VISION_BASE_URL` / `VISION_API_KEY` | OpenAI-compatible endpoint for the vision model (e.g. a self-hosted one). Unset → OpenAI. |
| `RAG_IMAGE_DIR` | Where extracted PDF figures are saved. |
| `RAG_MAX_RETRIES` | Max CRAG rewrite-retries on the full path. |
| `RAG_MIN_SCORE` | Minimum similarity for a passage to reach the planner. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters. |

## References

The design draws on the following work:

- **HyDE** — Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance
  Labels*, 2022. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496).
  Basis for the hypothetical-document query expansion in `query_gen.py`.
- **Multi-HyDE / agentic RAG** — Srinivasan et al., *Enhancing Financial RAG
  with Agentic AI and Multi-HyDE*, 2025.
  [arXiv:2509.16369](https://arxiv.org/abs/2509.16369). Basis for the multiple
  parallel HyDE queries in `query_gen.py`.
- **Question decomposition** — Ammann et al., *Question Decomposition for
  Retrieval-Augmented Generation*, 2025.
  [arXiv:2507.00355](https://arxiv.org/abs/2507.00355). Basis for splitting a
  question into sub-queries in `query_gen.py`.
- **Corrective RAG (CRAG)** — Yan et al., *Corrective Retrieval Augmented
  Generation*, 2024. [arXiv:2401.15884](https://arxiv.org/abs/2401.15884).
  Basis for the grade-and-retry self-correction in `correction.py`.
- **Multimodal RAG with visual citation** — Zhao et al., *FinRAGBench-V*, 2025.
  [arXiv:2505.17471](https://arxiv.org/abs/2505.17471). Motivates reading and
  citing PDF figures in `parsing.py` and `answer.py`.
- **Chart understanding** — Yi et al., *Multimodal Information Fusion for Chart
  Understanding: A Survey of MLLMs*, 2026.
  [arXiv:2602.10138](https://arxiv.org/abs/2602.10138). Background for figure
  and chart interpretation.
