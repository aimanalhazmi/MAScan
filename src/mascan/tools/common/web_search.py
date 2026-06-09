from typing import Any, ClassVar

from firecrawl import Firecrawl
from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for recent public-web information.")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of pages to return.")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the public web for recent information. "
        "Returns a list of matching pages with markdown body content."
    )
    input_schema: ClassVar[type[BaseModel] | None] = WebSearchInput

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Firecrawl.

        Accepts api_key explicitly, otherwise falls back to the
        FIRECRAWL_API_KEY environment variable.
        """
        self.api_key = api_key
        self.client: Firecrawl | None = None
        super().__init__()

    def run(self, query: str, max_results: int = 5, **_: Any) -> ToolResult[list[dict[str, Any]]]:
        try:
            results = self.search_impl(query=query, max_results=max_results)
            return ToolResult(
                success=True,
                data=results,
                source="web_search:firecrawl",
                metadata={"query": query, "count": len(results)},
            )
        except Exception as exc:
            self.logger.exception("web_search failed for query=%r", query)
            return ToolResult(
                success=False,
                source="web_search:firecrawl",
                error=str(exc),
            )

    def search_impl(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Executes the live search against Firecrawl's API endpoint.

        Returns a structured list containing titles, source URLs, and full
        scraped markdown strings tailored for LLM consumption.
        """
        if self.client is None:
            self.client = Firecrawl(api_key=self.api_key)

        response = self.client.search(query=query, limit=max_results)

        formatted_results = []
        for doc in response.web:
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
                    "markdown": markdown_content,
                }
            )

        return formatted_results
