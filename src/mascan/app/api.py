import json
from collections.abc import Iterator
from typing import Any


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from mascan.contracts import FinalReport
from mascan.core.exceptions import MAScanError
from mascan.core.logging import get_logger, configure_logging
from mascan.orchestrator import run as orchestrator_run
from mascan.orchestrator import stream as orchestrator_stream

import mascan.agents.economics  # noqa: F401
import mascan.agents.political  # noqa: F401
import mascan.agents.social # noqa: F401

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

    Each orchestrator node emits an SSE event when it completes. The
    final event contains the synthesized markdown.

    Useful for UIs that want to show progress (planning → agents → done)
    instead of waiting silently for the full result.
    """
    logger.info("Stream request: query=%r", request.query)

    def event_generator() -> Iterator[str]:
        try:
            yield sse_event({"event": "start", "query": request.query})

            # Stream orchestrator updates one node at a time.
            for chunk in orchestrator_stream(request.query):
                yield sse_event({"event": "node", **chunk})

            # Final event: explicit "done" so the client knows it's safe to close.
            yield sse_event({"event": "done"})
        except MAScanError as exc:
            logger.exception("Streaming failed")
            yield sse_event({"event": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()
