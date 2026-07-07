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

class ScholarSearchTool(BaseTool):
    name = "scholar_search"
    description = (
        "Search for academic papers and scholarly articles."
        "Returns a list of matching papers with metadata and abstracts."
    )

    input_schema: ClassVar[type[BaseModel] | None] = ScholarSearchInput

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


    def run(self, query: str, max_results: int = 5, **_: Any) -> ToolResult[list[dict[str, Any]]]:
        pass

    def search_literature(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Executes the live search against the Semantic Scholar API.

        Returns a structured list containing titles, authors, year, venue, abstracts, and URLs.
        """
        pass

    # Interaction with the Semantic Scholar API is rate-limited, so we need to ensure we don't exceed the allowed request rate.

    _lock = threading.Lock()
    _last_request = 0.0

    REQUEST_INTERVAL = 1.1  # Minimum interval between requests in seconds

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
        headers: dict[str, str] | None = _headers(),
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> requests.Response:
        """Makes a GET request to the Semantic Scholar API with retry logic."""

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
                time.sleep(2 ** attempt)  # Exponential backoff
