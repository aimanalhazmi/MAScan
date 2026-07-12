import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import mascan.agents.economics  # noqa: F401
import mascan.agents.environmental  # noqa: F401
import mascan.agents.legal  # noqa: F401
import mascan.agents.political  # noqa: F401
import mascan.agents.social  # noqa: F401
import mascan.agents.technological  # noqa: F401
from mascan.contracts import FinalReport
from mascan.contracts.retrieval import Citation, RagAnswer, RetrievalQuery, RetrievedChunk
from mascan.core.exceptions import ConfigError, MAScanError
from mascan.core.logging import configure_logging, get_logger
from mascan.orchestrator import resume as orchestrator_resume
from mascan.orchestrator import run as orchestrator_run
from mascan.orchestrator import stream as orchestrator_stream
from mascan.rag.answer import answer_question
from mascan.rag.ingest import ingest_file, ingest_text
from mascan.rag.retriever import get_retriever

configure_logging()
logger = get_logger("app.api")

class AnalyzeRequest(BaseModel):
    """Payload accepted by POST /analyze."""
    query: str = Field(..., min_length=1, description="The user's question.")

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"



def sse_event(payload: dict[str, Any]) -> str:
    """Format a dict as an SSE event line."""
    return f"data: {json.dumps(payload, default=pydantic_safe_default)}\n\n"


def pydantic_safe_default(obj: Any) -> Any:
    """JSON serializer fallback that knows how to handle Pydantic models."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def sse_from_events(events: Iterator[dict[str, Any]], thread_id: str) -> Iterator[str]:
    """Turn orchestrator node events into SSE lines. A clarification interrupt
    ends the stream so the client can collect an answer and POST /analyze/resume.
    """
    for ev in events:
        if ev["node"] == "__interrupt__":
            yield sse_event({"event": "clarification", "question": ev["question"], "thread_id": thread_id})
            return
        yield sse_event({"event": "node", **ev})
    yield sse_event({"event": "done"})

app = FastAPI(title="MAScan API", description="HTTP interface to the MAScan multi-agent orchestrator.", version="0.1.0")

@app.post("/analyze", response_model=FinalReport)
async def analyze(request: AnalyzeRequest) -> FinalReport:
    """Run the full orchestrator on the provided query."""
    logger.info("Analyze request: query=%r", request.query)
    try:
        report = orchestrator_run(request.query)
    except MAScanError as exc:
        logger.exception("Orchestrator failed")
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {exc}") from exc

    logger.info(
        "Analyze done: agents=%s failures=%s",
        sorted(report.agent_reports.keys()),
        sorted(report.failures.keys()),
    )
    return report

@app.post("/analyze/stream")
def analyze_stream(request: AnalyzeRequest) -> StreamingResponse:
    """Run the orchestrator with progressive Server-Sent Events output.

    Each orchestrator node emits an SSE event when it completes. If the planner
    needs clarification the stream ends with a `clarification` event carrying the
    question and thread_id; answer it via POST /analyze/resume. The final event
    is `done`.
    """
    logger.info("Stream request: query=%r", request.query)
    thread_id = str(uuid4())

    def event_generator() -> Iterator[str]:
        try:
            yield sse_event({"event": "start", "query": request.query, "thread_id": thread_id})
            yield from sse_from_events(orchestrator_stream(request.query, thread_id), thread_id)
        except Exception as exc:  # noqa: BLE001 - always close the stream with a reason
            logger.exception("Streaming failed")
            yield sse_event({"event": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


class ResumeRequest(BaseModel):
    """Payload for POST /analyze/resume: answer a pending clarification."""
    thread_id: str = Field(..., min_length=1, description="thread_id from the clarification event.")
    answer: str = Field(..., min_length=1, description="The user's clarification answer.")


@app.post("/analyze/resume")
def analyze_resume(request: ResumeRequest) -> StreamingResponse:
    """Resume a run paused by a clarification interrupt, streaming the rest."""
    logger.info("Resume request: thread=%s", request.thread_id)

    def event_generator() -> Iterator[str]:
        try:
            yield from sse_from_events(
                orchestrator_resume(request.thread_id, request.answer), request.thread_id
            )
        except Exception as exc:  # always close the stream with a reason
            logger.exception("Resume failed")
            yield sse_event({"event": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)
class IngestRequest(BaseModel):
    """Payload for POST /rag/ingest: plain text → chunked, embedded, stored."""
    text: str = Field(..., min_length=1, description="Raw text to ingest.")
    source: str = Field("api", description="Where the text entered from.")
    document: str = Field(..., min_length=1, description="Citation document id (e.g. filename/url).")


@app.post("/rag/ingest")
async def rag_ingest(request: IngestRequest) -> dict[str, int]:
    """Ingest plain text into the RAG store. Returns the number of chunks stored."""
    try:
        stored = await ingest_text(
            request.text, source=request.source, citation=Citation(document=request.document)
        )
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"stored": stored}


@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a .pdf, .md, or .txt file and ingest it. Returns chunks stored.

    PDFs are parsed page-by-page (text + figure images); md/txt go through the
    plain-text path. source is 'upload' so the generation step knows these
    chunks may carry visual elements.
    """
    document = file.filename or "upload"
    suffix = Path(document).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        stored = await ingest_file(tmp_path, document=document, source="upload")
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)
    return {"document": document, "stored": stored}


@app.post("/rag/search", response_model=list[RetrievedChunk])
async def rag_search(query: RetrievalQuery) -> list[RetrievedChunk]:
    """Dense similarity search over the RAG store. Empty list if RAG is disabled."""
    return await get_retriever().retrieve(query)


@app.post("/rag/answer", response_model=RagAnswer)
async def rag_answer(query: RetrievalQuery) -> RagAnswer:
    """Retrieve + generate a grounded answer with structured citations."""
    return await answer_question(query)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


static_dir = Path(os.environ.get("MASCAN_STATIC_DIR", Path(__file__).parent / "static"))
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
