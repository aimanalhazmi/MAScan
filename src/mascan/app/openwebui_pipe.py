"""Open WebUI Pipe: MAScan Analyst.

Install: Admin Settings -> Functions -> Add New Function -> paste, save, enable.
Adds the model "MAScan", which runs the full analyze flow.

Workflow (per message):
    1. Attached files are ingested into the RAG store (/rag/upload or /rag/ingest).
    2. The message text is analyzed by the orchestrator.
       - stream_progress = False: /analyze (blocking), returns the full report.
       - stream_progress = True:  /analyze/stream (SSE), yields progress then report.
"""

import asyncio
import inspect
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from pydantic import BaseModel, Field


def resolve(value: Any) -> Any:
    """Await Open WebUI's Files/Storage calls, which are async on newer versions.

    The pipe already runs inside an event loop, so run the coroutine in a
    throwaway thread rather than asyncio.run on the current loop.
    """
    if not inspect.iscoroutine(value):
        return value
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, value).result()


class Pipe:
    class Valves(BaseModel):
        mascan_api_url: str = Field(
            default="http://mascan-api:8000",
            description=(
                "Base URL of the MAScan API. Default works via docker compose; "
                "use http://host.docker.internal:8000 if Open WebUI is in Docker "
                "and MAScan runs on the host."
            ),
        )
        stream_progress: bool = Field(
            default=False,
            description="Stream per-agent progress instead of waiting for the full report.",
        )
        request_timeout_seconds: float = Field(default=180.0)

    def __init__(self) -> None:
        self.valves = self.Valves()

    def pipe(
        self,
        body: dict[str, Any],
        __files__: list[dict[str, Any]] | None = None,
        __request__: Any = None,
    ) -> str | Iterator[str]:
        prefix = ""
        if __files__:
            try:
                stored, names = self.ingest(__files__)
            except Exception as exc:  # noqa: BLE001
                return self.error(exc)
            prefix = f"Ingested {stored} chunks from {len(names)} file(s).\n\n"

        query = self.query(body)
        if not query:
            return prefix or "_(No query provided.)_"

        if self.valves.stream_progress:
            return self.streaming(query, prefix)
        return prefix + self.blocking(query)

    def blocking(self, query: str) -> str:
        url = f"{self.valves.mascan_api_url.rstrip('/')}/analyze"
        try:
            with httpx.Client(timeout=self.valves.request_timeout_seconds) as client:
                resp = client.post(url, json={"query": query})
                resp.raise_for_status()
                report = resp.json()
        except Exception as exc:  # noqa: BLE001
            return self.error(exc)
        return self.report(report)

    def streaming(self, query: str, prefix: str = "") -> Iterator[str]:
        if prefix:
            yield prefix
        url = f"{self.valves.mascan_api_url.rstrip('/')}/analyze/stream"
        try:
            with httpx.Client(timeout=self.valves.request_timeout_seconds) as client:
                with client.stream("POST", url, json={"query": query}) as resp:
                    resp.raise_for_status()
                    yield from self.consume(resp)
        except Exception as exc:  # noqa: BLE001
            yield self.error(exc)

    def consume(self, resp: httpx.Response) -> Iterator[str]:
        """Turn SSE events into progress lines, then the final report."""
        final = None
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[len("data: ") :])
            except json.JSONDecodeError:
                continue

            kind = event.get("event")
            if kind == "start":
                yield "Starting analysis...\n\n"
            elif kind == "node":
                update = event.get("update") or {}
                if event.get("node") == "synthesizer":
                    final = update.get("final_markdown") or update.get("final_summary")
                status = self.status(event.get("node", "?"), update)
                if status:
                    yield status + "\n\n"
            elif kind == "done":
                yield f"---\n\n{final}" if final else "_(No final report produced.)_"
            elif kind == "error":
                yield f"\n\nError: {event.get('message', 'Unknown error')}"

    @staticmethod
    def status(node: str, update: dict[str, Any]) -> str:
        if node == "planner":
            plan = update.get("plan") or {}
            if plan:
                return f"Planner selected: {', '.join(sorted(plan.keys()))}"
            return "Planner ran but selected no agents."
        if node == "synthesizer":
            return "Synthesizing final report..."
        if node == "validator":
            return "Validating final report..."

        reports = update.get("reports") or {}
        if node in reports:
            report = reports[node]
            markdown = report.get("rendered_markdown") or report.get("findings") or ""
            return (
                f"<details>\n<summary>{node} finished — click to expand</summary>\n\n"
                f"{markdown}\n\n</details>"
            )
        failures = update.get("failures") or {}
        if node in failures:
            return f"**{node}** failed: `{failures.get(node)}`"
        return ""

    def ingest(self, files: list[dict[str, Any]]) -> tuple[int, list[str]]:
        """Push attached files into the RAG store.

        Raw bytes go to /rag/upload so PDFs get parsed; when Open WebUI's file
        modules are unavailable, fall back to the pre-extracted text via /rag/ingest.
        """
        try:
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage
        except Exception:  # noqa: BLE001
            Files = Storage = None  # noqa: N806

        api = self.valves.mascan_api_url.rstrip("/")
        stored = 0
        names: list[str] = []
        with httpx.Client(timeout=self.valves.request_timeout_seconds) as client:
            for f in files:
                file_id = f.get("id") or f.get("file", {}).get("id")
                if Files and Storage and file_id:
                    meta = resolve(Files.get_file_by_id(file_id))
                    with open(resolve(Storage.get_file(meta.path)), "rb") as fh:
                        resp = client.post(
                            f"{api}/rag/upload",
                            files={"file": (meta.filename, fh.read())},
                        )
                    name = meta.filename
                else:
                    info = f.get("file", {})
                    content = (info.get("data") or {}).get("content", "")
                    if not content:
                        continue
                    name = info.get("filename") or f.get("name") or "upload"
                    resp = client.post(
                        f"{api}/rag/ingest",
                        json={"text": content, "source": "upload", "document": name},
                    )
                resp.raise_for_status()
                stored += resp.json().get("stored", 0)
                names.append(name)
        return stored, names

    @staticmethod
    def query(body: dict[str, Any]) -> str:
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = (p.get("text", "") for p in content if isinstance(p, dict))
                return " ".join(parts).strip()
        return ""

    @staticmethod
    def report(report: dict[str, Any]) -> str:
        markdown = report.get("rendered_markdown") or report.get("summary") or ""
        if not markdown:
            return "_(MAScan returned an empty report.)_"
        failures = report.get("failures") or {}
        if failures:
            lines = "\n".join(f"- **{name}**: {err}" for name, err in failures.items())
            markdown += f"\n\n---\n\nSome agents failed:\n{lines}"
        return markdown

    def error(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPError):
            return (
                f"Could not reach the MAScan API at `{self.valves.mascan_api_url}`.\n\n"
                f"Error: `{exc}`\n\n"
                "Check the API is running (`make run-api`) and the URL in this Pipe's "
                "settings is correct. From inside Docker, use `http://host.docker.internal:8000`."
            )
        return f"Unexpected error talking to MAScan: `{exc}`"
