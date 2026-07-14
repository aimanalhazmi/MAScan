"""Fetch the exact public URL cited by a generated report."""

import ipaddress
import socket
from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import urlsplit

from firecrawl import Firecrawl
from pydantic import BaseModel, Field
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool


class SourceFetchInput(BaseModel):
    url: str = Field(..., description="Exact public HTTP or HTTPS URL to fetch.")


class UnsafeSourceUrlError(ValueError):
    """Raised when a URL could target a non-public network resource."""


class SourceFetchTool(BaseTool):
    """Retrieve a cited page through the configured Firecrawl service."""

    name = "source_fetch"
    description = "Fetch the readable Markdown content of an exact public citation URL."
    input_schema: ClassVar[type[BaseModel] | None] = SourceFetchInput

    TIMEOUT_MS: ClassVar[int] = 30_000
    MAX_MARKDOWN_CHARS: ClassVar[int] = 100_000

    def __init__(self, api_key: str | None = None, api_url: str | None = None) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.client: Firecrawl | None = None
        super().__init__()

    def run(self, url: str, **_: Any) -> ToolResult[dict[str, Any]]:
        checked_at = datetime.now(UTC).isoformat()
        try:
            self._validate_public_url(url)
            document = self._fetch_with_retries(url)
            metadata = getattr(document, "metadata", None) or {}
            if not isinstance(metadata, dict):
                metadata = {}

            markdown = getattr(document, "markdown", "") or ""
            if not isinstance(markdown, str) or not markdown.strip():
                raise ValueError("Firecrawl returned no readable page content")

            title = getattr(document, "title", "") or metadata.get("title") or "Untitled source"
            final_url = getattr(document, "url", "") or metadata.get("url") or url
            self._validate_public_url(str(final_url))

            truncated = len(markdown) > self.MAX_MARKDOWN_CHARS
            if truncated:
                markdown = markdown[: self.MAX_MARKDOWN_CHARS].rstrip()

            data = {
                "requested_url": url,
                "final_url": str(final_url),
                "title": str(title),
                "markdown": markdown,
                "checked_at": checked_at,
                "truncated": truncated,
            }
            return ToolResult(
                success=True,
                data=data,
                source="source_fetch:firecrawl",
                metadata={"checked_at": checked_at, "truncated": truncated},
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("source_fetch failed for url=%r: %s", url, exc)
            return ToolResult(
                success=False,
                source="source_fetch:firecrawl",
                error=f"{type(exc).__name__}: {exc}",
                metadata={"url": url, "checked_at": checked_at},
            )

    def _fetch_with_retries(self, url: str) -> Any:
        if self.client is None:
            self.client = Firecrawl(
                api_key=self.api_key or "self-hosted",
                api_url=self.api_url,
            )

        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception(self._is_retryable),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self.client.scrape(
                    url,
                    formats=["markdown"],
                    only_main_content=True,
                    timeout=self.TIMEOUT_MS,
                )
        raise RuntimeError("Firecrawl retry loop ended without a result")

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        message = str(exc).lower()
        non_retryable = ("400", "401", "403", "404", "invalid url", "unsupported url")
        return not any(marker in message for marker in non_retryable)

    @classmethod
    def _validate_public_url(cls, url: str) -> None:
        try:
            parsed = urlsplit(url.strip())
        except ValueError as exc:
            raise UnsafeSourceUrlError(f"Invalid URL: {exc}") from exc

        if parsed.scheme.lower() not in {"http", "https"}:
            raise UnsafeSourceUrlError("Only HTTP and HTTPS URLs are allowed")
        if parsed.username or parsed.password:
            raise UnsafeSourceUrlError("URLs containing credentials are not allowed")
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if not hostname:
            raise UnsafeSourceUrlError("URL has no hostname")
        if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
            raise UnsafeSourceUrlError("Local hostnames are not allowed")

        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            if "." not in hostname:
                raise UnsafeSourceUrlError("Single-label hostnames are not allowed") from None
            try:
                addresses = {
                    item[4][0]
                    for item in socket.getaddrinfo(
                        hostname,
                        parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
                        type=socket.SOCK_STREAM,
                    )
                }
            except socket.gaierror as exc:
                raise UnsafeSourceUrlError(f"Hostname could not be resolved: {hostname}") from exc
            if not addresses:
                raise UnsafeSourceUrlError(
                    f"Hostname could not be resolved: {hostname}"
                ) from None
            for address in addresses:
                if not ipaddress.ip_address(address).is_global:
                    raise UnsafeSourceUrlError(
                        "Private or non-global network addresses are not allowed"
                    ) from None
        else:
            if not literal.is_global:
                raise UnsafeSourceUrlError(
                    "Private or non-global network addresses are not allowed"
                )
