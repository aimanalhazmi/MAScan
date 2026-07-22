from typing import Any, ClassVar
from urllib.parse import urlparse

from firecrawl import Firecrawl
from firecrawl.v2.types import ScrapeOptions
from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for recent public-web information.")
    max_results: int = Field(5, description="Requested pages; values are clamped to 1–5.")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the public web for recent information. "
        "Returns a list of matching pages with markdown body content."
    )
    input_schema: ClassVar[type[BaseModel] | None] = WebSearchInput

    MAX_RESULTS: ClassVar[int] = 5
    # Firecrawl returns the full scraped page as markdown. Left unbounded, a
    # handful of long pages overflows the model context window, so cap each
    # page's body before it is handed to the LLM.
    #
    # 4000 is measured, not guessed. Cutting it to 1200 to test whether page
    # verbosity was crowding out PESTEL factor coverage made every headline
    # metric worse over the same 25 cases (categorization -0.060, depth -0.088,
    # combined -0.045) and erased MAScan's significant edge over zero-shot
    # (+0.045 p=0.026 -> +0.001 p=0.71). Only grounding improved (+0.063).
    # Do not lower this without re-running the 25-case comparison.
    MAX_MARKDOWN_CHARS: ClassVar[int] = 4000
    # Firecrawl only scrapes result pages when search is given scrape_options.
    # Without it every result comes back with an empty markdown body, so the
    # agent receives link titles and no page content at all.
    SCRAPE_TIMEOUT_MS: ClassVar[int] = 20000

    def __init__(self, api_key: str | None = None, api_url: str | None = None) -> None:
        """Initialize Firecrawl.

        Accepts api_key/api_url explicitly, otherwise falls back to the
        FIRECRAWL_API_KEY / FIRECRAWL_API_URL environment variables.
        Set api_url to point at a self-hosted Firecrawl instead of the cloud.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.client: Firecrawl | None = None
        super().__init__()

    def run(self, query: str, max_results: int = 5, **_: Any) -> ToolResult[list[dict[str, Any]]]:
        try:
            bounded_results = max(1, min(max_results, self.MAX_RESULTS))
            results = self.search_impl(query=query, max_results=bounded_results)
            return ToolResult(
                success=True,
                data=results,
                source="web_search:firecrawl",
                metadata={
                    "query": query,
                    "count": len(results),
                    "limit_applied": max_results != bounded_results,
                },
            )
        except Exception as exc:
            self.logger.exception("web_search failed for query=%r", query)
            return ToolResult(
                success=False,
                source="web_search:firecrawl",
                error=str(exc),
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

    def search_impl(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Executes the live search against Firecrawl's API endpoint.

        Returns a structured list containing titles, source URLs, and full
        scraped markdown strings tailored for LLM consumption.
        """
        if self.client is None:
            self.client = self._build_firecrawl_client()

        bounded_results = max(1, min(max_results, self.MAX_RESULTS))
        try:
            response = self.client.search(
                query=query,
                limit=bounded_results,
                scrape_options=ScrapeOptions(
                    formats=["markdown"],
                    only_main_content=True,
                    timeout=self.SCRAPE_TIMEOUT_MS,
                ),
            )
        except AttributeError as exc:
            # None response when the HTTP request fails without a response obj
            raise ConnectionError(f"Firecrawl search failed (likely network/server error): {exc}") from exc

        formatted_results = []
        for doc in (getattr(response, "web", None) or [])[:bounded_results]:
            # Firecrawl returns metadata as a plain dict for unscraped hits and as
            # a DocumentMetadata model once scraping is enabled, so read both. The
            # scraped shape often carries the only usable url/title.
            url = self._metadata_value(doc, "url") or getattr(doc, "url", "") or ""
            title = self._metadata_value(doc, "title") or getattr(doc, "title", None)
            markdown_content = getattr(doc, "markdown", "")

            formatted_results.append(
                {
                    # Fall back to the domain (not a literal "No Title") when a page
                    # exposes no title, so citations stay meaningful.
                    "title": title or self._domain_title(url),
                    "url": url,
                    "markdown": self._truncate_markdown(markdown_content),
                }
            )

        return formatted_results

    @staticmethod
    def _domain_title(url: str) -> str:
        return urlparse(url).netloc.removeprefix("www.") or "Untitled source"

    @staticmethod
    def _metadata_value(doc: Any, key: str) -> str | None:
        """Read one metadata field whether it is a dict or a pydantic model."""
        metadata = getattr(doc, "metadata", None)
        if not metadata:
            return None
        value = (
            metadata.get(key)
            if isinstance(metadata, dict)
            else getattr(metadata, key, None)
        )
        return str(value) if value else None

    @classmethod
    def _truncate_markdown(cls, markdown: str) -> str:
        """Cap a single page's markdown so a few long pages can't overflow context."""
        truncated: str = cls.truncate_text(
            markdown,
            cls.MAX_MARKDOWN_CHARS,
            marker="\n\n[...truncated...]",
        )
        return truncated
