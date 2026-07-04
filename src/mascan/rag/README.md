# RAG

Retrieval-Augmented Generation for MAScan. It ingests documents (text and
PDFs), stores them as embeddings, retrieves the relevant passages for a
question, and generates an answer grounded in those passages with source
citations.

RAG is optional. Without `DATABASE_URL` set, retrieval returns nothing
(`NullRetriever`) and the rest of MAScan runs unchanged.

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

Agents can search the index through the `rag_search` tool
(`mascan/tools/common/rag_search.py`) by listing it in their `config.yaml`.

HTTP endpoints (see `mascan/app/api.py`): `POST /rag/ingest`, `/rag/upload`,
`/rag/search`, `/rag/answer`.

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
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters. |
