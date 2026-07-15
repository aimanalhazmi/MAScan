from typing import Any, ClassVar

from firecrawl import Firecrawl
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
    MAX_MARKDOWN_CHARS: ClassVar[int] = 4000

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
            response = self.client.search(query=query, limit=bounded_results)
        except AttributeError as exc:
            # None response when the HTTP request fails without a response obj
            raise ConnectionError(f"Firecrawl search failed (likely network/server error): {exc}") from exc

        formatted_results = []
        for doc in (getattr(response, "web", None) or [])[:bounded_results]:
            title = getattr(doc, "title", "No Title")
            url = getattr(doc, "url", "")

            markdown_content = getattr(doc, "markdown", "")

            if hasattr(doc, "metadata") and doc.metadata:
                title = doc.metadata.get("title", title)
                url = doc.metadata.get("url", url)

            formatted_results.append(
                {
                    "title": title,
                    "url": url,
                    "markdown": self._truncate_markdown(markdown_content),
                }
            )

        return formatted_results

    @classmethod
    def _truncate_markdown(cls, markdown: str) -> str:
        """Cap a single page's markdown so a few long pages can't overflow context."""
        return cls.truncate_text(
            markdown,
            cls.MAX_MARKDOWN_CHARS,
            marker="\n\n[...truncated...]",
        )
