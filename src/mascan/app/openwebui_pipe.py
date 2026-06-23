"""Open WebUI Pipe Function: MAScan Analyst.

Paste into Open WebUI:
    Admin Settings → Functions → Add New Function
    Paste this file, save, enable.

A new model "MAScan" appears in the model dropdown.

This Pipe supports TWO modes, switchable via Valves in the admin UI:

  - stream_progress = False (default)
      Calls /analyze (blocking). Open WebUI shows a typing indicator
      while waiting (~10-30s), then displays the full markdown report.

  - stream_progress = True
      Calls /analyze/stream (SSE). Yields short status updates as each
      orchestrator node finishes ("Planning...", ..., "Economics done"),
      then yields the final markdown report.
"""


import json
from collections.abc import Iterator
from typing import Any

import httpx
from pydantic import BaseModel, Field


class Pipe:
    """The Pipe class Open WebUI looks for when loading this file."""

    class Valves(BaseModel):
        """User-configurable settings shown in the Open WebUI admin panel."""

        mascan_api_url: str = Field(
            default="http://mascan-api:8000",
            description=(
                "Base URL of the MAScan FastAPI server. "
                "Default (http://mascan-api:8000) works when running via "
                "docker compose. "
                "Use http://host.docker.internal:8000 if Open WebUI is in Docker "
                "and MAScan runs on the host. "
                "Use http://localhost:8000 only if both run on the host."
            ),
        )
        stream_progress: bool = Field(
            default=False,
            description=(
                "If true, stream progress updates as each agent finishes. "
                "If false, show a typing indicator until the full report is ready."
            ),
        )
        request_timeout_seconds: float = Field(
            default=180.0,
            description="HTTP timeout for the API call. Increase for slow LLMs.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    def pipe(self, body: dict[str, Any]) -> str | Iterator[str]:
        """Open WebUI entry point. Returns either a string (blocking) or
        an iterator of strings (streaming), based on the Valves toggle.
        """
        query = self.extract_last_user_message(body)
        if not query:
            return "_(No query provided.)_"

        if self.valves.stream_progress:
            return self.pipe_streaming(query)
        return self.pipe_blocking(query)

    # Mode A: Calling /analyze (blocking)

    def pipe_blocking(self, query: str) -> str:
        """Call /analyze and return the full markdown report."""
        try:
            report = self.call_analyze(query)
        except httpx.HTTPError as exc:
            return self.format_http_error(exc)
        except Exception as exc:
            return f"⚠️ Unexpected error talking to MAScan: `{exc}`"
        return self.format_report(report)

    def call_analyze(self, query: str) -> dict[str, Any]:
        url = f"{self.valves.mascan_api_url.rstrip('/')}/analyze"
        with httpx.Client(timeout=self.valves.request_timeout_seconds) as client:
            response = client.post(url, json={"query": query})
            response.raise_for_status()
            return response.json()

    # Mode B: Calling /analyze/stream

    def pipe_streaming(self, query: str) -> Iterator[str]:
        """Call /analyze/stream and yield progress updates plus the final report."""
        url = f"{self.valves.mascan_api_url.rstrip('/')}/analyze/stream"
        try:
            with httpx.Client(timeout=self.valves.request_timeout_seconds) as client:
                with client.stream("POST", url, json={"query": query}) as response:
                    response.raise_for_status()
                    yield from self.consume_sse(response)
        except httpx.HTTPError as exc:
            yield self.format_http_error(exc)
        except Exception as exc:  # noqa: BLE001
            yield f"\n\n⚠️ Unexpected error talking to MAScan: `{exc}`"

    def consume_sse(self, response: httpx.Response) -> Iterator[str]:
        """Parse SSE 'data: ...' lines and yield human-readable progress.

        After the final report node events, yields the rendered markdown so the
        full report appears as the last message chunk.
        """
        final_markdown: str | None = None

        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: ") :])
            except json.JSONDecodeError:
                continue

            event_type = payload.get("event")

            if event_type == "start":
                yield "⏳ Starting analysis...\n\n"

            elif event_type == "node":
                node = payload.get("node", "?")
                update = payload.get("update") or {}
                comment = self.comment_for_node(node, update)
                if comment:
                    yield comment + "\n\n"
                # Capture report markdown as it evolves; validator appends Fact Check.
                if node in {"synthesizer", "validator"}:
                    final_markdown = update.get("final_markdown") or update.get("final_summary")

            elif event_type == "done":
                if final_markdown:
                    yield "---\n\n"
                    yield final_markdown
                else:
                    yield "_(No final report produced.)_"

            elif event_type == "error":
                yield f"\n\n⚠️ {payload.get('message', 'Unknown error')}"

    @staticmethod
    def comment_for_node(node: str, update: dict[str, Any]) -> str:
        """Produce a short, friendly status line for a node completion."""
        if node == "planner":
            plan = update.get("plan") or {}
            if plan:
                names = ", ".join(sorted(plan.keys()))
                return f"📋 Planner selected: {names}"
            return "📋 Planner ran but selected no agents."
        if node == "synthesizer":
            return "🧩 Synthesizing final report..."
        if node == "validator":
            return "🔎 Validating final report..."
        # Otherwise it's an agent node.
        reports = update.get("reports") or {}
        failures = update.get("failures") or {}
        if node in reports:
            report = reports[node]
            agent_markdown = report.get("rendered_markdown") or report.get("findings") or ""
            confidence = report.get("confidence")
            conf_str = f" (confidence {confidence:.2f})" if isinstance(confidence, (int, float)) else ""

            # Collapsible block — click the summary to expand.
            return (
                f"<details>\n"
                f"<summary>✅ {node} finished{conf_str} — click to expand</summary>\n\n"
                f"{agent_markdown}\n\n"
                f"</details>"
            )

        if node in failures:
            return f"❌ **{node}** failed: `{failures.get(node)}`"

        return ""

    # --- Shared helpers

    @staticmethod
    def extract_last_user_message(body: dict[str, Any]) -> str:
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    return " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ).strip()
        return ""

    @staticmethod
    def format_report(report: dict[str, Any]) -> str:
        markdown = report.get("rendered_markdown") or report.get("summary") or ""
        if not markdown:
            return "_(MAScan returned an empty report.)_"
        failures = report.get("failures") or {}
        if failures:
            failure_lines = "\n".join(f"- **{name}**: {err}" for name, err in failures.items())
            markdown += f"\n\n---\n\n⚠️ Some agents failed:\n{failure_lines}"
        return markdown

    def format_http_error(self, exc: httpx.HTTPError) -> str:
        url = self.valves.mascan_api_url
        return (
            f"⚠️ Could not reach the MAScan API at `{url}`.\n\n"
            f"Error: `{exc}`\n\n"
            "Check that the API is running (`make run-api`) and that the URL "
            "in this Pipe's settings is correct. From inside Docker, use "
            "`http://host.docker.internal:8000`."
        )
