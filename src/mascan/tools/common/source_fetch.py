"""Fetch the exact public web URL cited by a generated report."""

import ipaddress
import socket
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin, urlsplit

import httpx
from firecrawl import Firecrawl
from pydantic import BaseModel, Field
from pypdf import PdfReader
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool


class SourceFetchInput(BaseModel):
    url: str = Field(..., description="Exact public HTTP or HTTPS URL to fetch.")


class UnsafeSourceUrlError(ValueError):
    """Raised when a URL could target a non-public network resource."""


class InaccessibleSourceError(RuntimeError):
    """Raised when a source is confirmed missing or invalid."""


class OperationalSourceFetchError(RuntimeError):
    """Raised when a source may exist but could not be read by the validator."""


class _ReadableHTMLParser(HTMLParser):
    """Small dependency-free HTML-to-text fallback for source validation."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._skip_depth:
            return
        self.parts.append(text)
        if self._in_title:
            self.title_parts.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.parts)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts)


class SourceFetchTool(BaseTool):
    """Retrieve a cited page through Firecrawl with a direct HTTP fallback."""

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
        except UnsafeSourceUrlError as exc:
            return self._failure(url, checked_at, exc, failure_kind="inaccessible")

        try:
            try:
                document = self._fetch_with_retries(url)
                data = self._firecrawl_data(document, url, checked_at)
            except Exception as firecrawl_exc:  # noqa: BLE001
                self.logger.info("Firecrawl failed for %r; trying direct fetch", url)
                try:
                    data = self._fetch_direct(url, checked_at)
                except InaccessibleSourceError as exc:
                    return self._failure(url, checked_at, exc, failure_kind="inaccessible")
                except Exception as direct_exc:  # noqa: BLE001
                    error = OperationalSourceFetchError(
                        "Firecrawl and direct retrieval failed: "
                        f"Firecrawl={type(firecrawl_exc).__name__}: {firecrawl_exc}; "
                        f"direct={type(direct_exc).__name__}: {direct_exc}"
                    )
                    return self._failure(url, checked_at, error, failure_kind="operational")
            return ToolResult(
                success=True,
                data=data,
                source=f"source_fetch:{data.get('retrieval_method', 'firecrawl')}",
                metadata={
                    "checked_at": checked_at,
                    "truncated": bool(data.get("truncated")),
                    "retrieval_method": data.get("retrieval_method", "firecrawl"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return self._failure(url, checked_at, exc, failure_kind="operational")

    def _firecrawl_data(
        self,
        document: Any,
        requested_url: str,
        checked_at: str,
    ) -> dict[str, Any]:
        metadata = getattr(document, "metadata", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        markdown = getattr(document, "markdown", "") or ""
        if not isinstance(markdown, str) or not markdown.strip():
            raise OperationalSourceFetchError("Firecrawl returned no readable page content")
        title = getattr(document, "title", "") or metadata.get("title") or "Untitled source"
        final_url = str(getattr(document, "url", "") or metadata.get("url") or requested_url)
        self._validate_public_url(final_url)
        markdown, truncated = self._truncate(markdown)
        return {
            "requested_url": requested_url,
            "final_url": final_url,
            "title": str(title),
            "markdown": markdown,
            "checked_at": checked_at,
            "truncated": truncated,
            "retrieval_method": "firecrawl",
        }

    def _fetch_direct(self, url: str, checked_at: str) -> dict[str, Any]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.8",
        }
        current_url = url
        with httpx.Client(
            headers=headers,
            timeout=self.TIMEOUT_MS / 1000,
            follow_redirects=False,
        ) as client:
            for _ in range(6):
                self._validate_public_url(current_url)
                response = client.get(current_url)
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise OperationalSourceFetchError(
                            "Redirect response had no Location header"
                        )
                    current_url = urljoin(current_url, location)
                    continue
                break
            else:
                raise OperationalSourceFetchError("Too many redirects")

        if response.status_code in {404, 410}:
            raise InaccessibleSourceError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise OperationalSourceFetchError(f"HTTP {response.status_code}")

        content_type = response.headers.get("content-type", "").lower()
        title = "Untitled source"
        if "application/pdf" in content_type or current_url.lower().endswith(".pdf"):
            try:
                reader = PdfReader(BytesIO(response.content))
                markdown = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:  # noqa: BLE001
                raise OperationalSourceFetchError(f"PDF text extraction failed: {exc}") from exc
        elif "html" in content_type or not content_type:
            parser = _ReadableHTMLParser()
            parser.feed(response.text)
            markdown = parser.text
            title = parser.title or title
            lowered = markdown.lower()
            challenge_markers = (
                "verify you are human",
                "captcha",
                "access denied",
                "enable javascript",
                "cloudflare ray id",
            )
            if any(marker in lowered for marker in challenge_markers):
                raise OperationalSourceFetchError("Direct response was an anti-bot challenge page")
        else:
            markdown = response.text

        if not markdown.strip():
            raise OperationalSourceFetchError("Direct retrieval returned no readable content")
        markdown, truncated = self._truncate(markdown)
        return {
            "requested_url": url,
            "final_url": current_url,
            "title": title,
            "markdown": markdown,
            "checked_at": checked_at,
            "truncated": truncated,
            "retrieval_method": "direct",
        }

    def _truncate(self, markdown: str) -> tuple[str, bool]:
        truncated = len(markdown) > self.MAX_MARKDOWN_CHARS
        if truncated:
            markdown = markdown[: self.MAX_MARKDOWN_CHARS].rstrip()
        return markdown, truncated

    def _failure(
        self,
        url: str,
        checked_at: str,
        exc: BaseException,
        *,
        failure_kind: Literal["inaccessible", "operational"],
    ) -> ToolResult[dict[str, Any]]:
        self.logger.warning("source_fetch failed for url=%r: %s", url, exc)
        return ToolResult(
            success=False,
            source="source_fetch",
            error=f"{type(exc).__name__}: {exc}",
            metadata={
                "url": url,
                "checked_at": checked_at,
                "failure_kind": failure_kind,
            },
        )

    def _build_firecrawl_client(self) -> Firecrawl:
        """Create a Firecrawl client for cloud or self-hosted mode."""
        if self.api_url:
            # Self-hosted Firecrawl needs no key; pass a placeholder so the SDK
            # doesn't reject the empty value.
            return Firecrawl(
                api_key=self.api_key or "self-hosted",
                api_url=self.api_url,
            )
        if not self.api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY is required when FIRECRAWL_API_URL is not set."
            )
        # Cloud mode: omit api_url entirely. Passing api_url=None crashes the SDK.
        return Firecrawl(api_key=self.api_key)

    def _fetch_with_retries(self, url: str) -> Any:
        if self.client is None:
            self.client = self._build_firecrawl_client()

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
