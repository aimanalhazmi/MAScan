"""Scholar Search Tool"""

import threading
import time
import requests

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool

class ScholarSearchInput(BaseModel):
    query: str = Field(..., description="Search query for academic papers and scholarly articles.")
    max_results: int = Field(5, description="Maximum number of results to return.")
    year_from: int | None = Field(None, description="Filter for the start of the publication year range.")
    year_to: int | None = Field(None, description="Filter for the end of the publication year range.")

class ScholarSearchTool(BaseTool):
    name = "scholar_search"
    description = (
        "Search for academic papers and scholarly articles."
        "Returns a list of matching papers with metadata and abstracts."
    )

    input_schema: ClassVar[type[BaseModel] | None] = ScholarSearchInput
    DEFAULT_API_URL: ClassVar[str] = "https://api.semanticscholar.org/graph/v1"

    DEFAULT_FIELDS = [
        "title",
        "authors",
        "year",
        "venue",
        "abstract",
        "url",
    ]

    def __init__(self, api_key: str | None = None, api_url: str | None = None) -> None:
        """Initialize the Scholar Search Tool.

        API key and URL can be provided explicitly, otherwise they will be read from environment variables.
        API key is optional but recommended for better rate limits and access to certain endpoints.
        """
        self.api_key = api_key
        self.api_url = api_url
        super().__init__()


    def run(self, query: str, max_results: int = 5, year_from: int | None = None, year_to: int | None = None) -> ToolResult[list[dict[str, Any]]]:
        try:
            results = self.search_literature(query=query, max_results=max_results, year_from=year_from, year_to=year_to)
            return ToolResult(
                success=True,
                data=results,
                source="scholar_search:semantic_scholar",
                metadata={"query": query, "count": len(results)},
            )
        except Exception as exc:
            self.logger.exception("scholar_search failed for query=%r", query)
            return ToolResult(
                success=False,
                source="scholar_search:semantic_scholar",
                error=str(exc),
            )

    def search_literature(self, query: str, max_results: int, year_from: int | None = None, year_to: int | None = None) -> list[dict[str, Any]]:
        """
        Relevance-ranked search for academic papers and scholarly articles using the Semantic Scholar API.

        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return.
            year_from (int | None): Optional filter for the start of the publication year range.
            year_to (int | None): Optional filter for the end of the publication year range.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing paper metadata and abstracts.
        """
        params = {
            "query": query,
            "limit": max_results,
            "fields": ",".join(self.DEFAULT_FIELDS),
        }

        if year_from is not None and year_to is not None:
            params["year"] = f"{year_from}-{year_to}"
        elif year_from is not None:
            params["year"] = f"{year_from}-"
        elif year_to is not None:
            params["year"] = f"-{year_to}"

        base_url = self.api_url or self.DEFAULT_API_URL
        response = self._request(url=f"{base_url}/paper/search", params=params)
        return response.json().get("data", [])


    # Interaction with the Semantic Scholar API is rate-limited, so we need to ensure we don't exceed the allowed request rate.

    _lock = threading.Lock()
    _last_request = 0.0

    REQUEST_INTERVAL = 1.2  # Minimum interval between requests (Semantic Scholar allows 1/sec cumulative)

    def _wait_for_slot(self):
        """Waits for the next available request slot based on the rate limit."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request

            if elapsed < self.REQUEST_INTERVAL:
                time.sleep(self.REQUEST_INTERVAL - elapsed)

            self._last_request = time.monotonic()

    def _headers(self) -> dict[str, str]:
        """Returns the headers for the API request, including the API key if provided."""
        headers = {}

        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _request(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> requests.Response:
        """Makes a GET request to the Semantic Scholar API with retry logic."""

        headers = headers or self._headers()

        for attempt in range(max_retries):
            self._wait_for_slot()
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    raise
                # Retry-After on 429, else exponential backoff
                retry_after = getattr(e.response, "headers", {}).get("Retry-After") if e.response is not None else None
                time.sleep(float(retry_after) if retry_after else 2 ** attempt)
