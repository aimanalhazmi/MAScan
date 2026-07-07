"""Open WebUI Pipe: MAScan RAG.

Install: Admin Settings -> Functions -> Add New Function -> paste, save, enable.
Adds the model "MAScan RAG", which talks only to the RAG endpoints.

Workflow (per message):
    1. Attached files are ingested into the RAG store (/rag/upload or /rag/ingest).
    2. If the message has text, it is answered from the store (/rag/answer).
    3. The reply combines the ingest summary and the grounded answer + sources.
"""

import asyncio
import inspect
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
        k: int = Field(default=5, description="Number of chunks to retrieve.")
        request_timeout_seconds: float = Field(default=180.0)

    def __init__(self) -> None:
        self.valves = self.Valves()

    def pipe(
        self,
        body: dict[str, Any],
        __files__: list[dict[str, Any]] | None = None,
        __request__: Any = None,
    ) -> str:
        prefix = ""
        if __files__:
            try:
                stored, names = self.ingest(__files__)
            except Exception as exc:  # noqa: BLE001
                return self.error(exc)
            prefix = f"Ingested {stored} chunks from: {', '.join(names)}\n\n"

        query = self.query(body)
        if not query:
            return prefix or "_(No query provided.)_"
        try:
            return prefix + self.answer(query)
        except Exception as exc:  # noqa: BLE001
            return self.error(exc)

    def answer(self, query: str) -> str:
        api = self.valves.mascan_api_url.rstrip("/")
        with httpx.Client(timeout=self.valves.request_timeout_seconds) as client:
            resp = client.post(f"{api}/rag/answer", json={"query": query, "k": self.valves.k})
            resp.raise_for_status()
            data = resp.json()

        answer = data.get("answer") or "_(No answer produced.)_"
        citations = data.get("citations") or []
        if not citations:
            return answer

        sources = []
        for c in citations:
            page = c.get("page")
            sources.append(f"- {c.get('document', '?')}" + (f", p. {page}" if page else ""))
        return answer + "\n\n**Sources**\n" + "\n".join(sources)

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

    def error(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPError):
            return (
                f"Could not reach the MAScan API at `{self.valves.mascan_api_url}`.\n\n"
                f"Error: `{exc}`\n\n"
                "Check the API is running and the URL in this Pipe's settings is "
                "correct. From inside Docker, use `http://host.docker.internal:8000`."
            )
        return f"Unexpected error talking to MAScan: `{exc}`"
