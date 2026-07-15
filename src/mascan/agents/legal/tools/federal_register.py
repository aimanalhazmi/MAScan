"""FederalRegisterTool — search the U.S. Federal Register.

The Federal Register is the official daily journal of the U.S. government:
rules, proposed rules, notices, and presidential documents. The API is public
and requires no API key.

Docs: https://www.federalregister.gov/developers/documentation/api/v1
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.http_client import http_get


class FederalRegisterInput(BaseModel):
    query: str
    per_page: int = Field(5, description="Requested documents; values are clamped to 1–10.")
    order: str = "newest"


class FederalRegisterTool(BaseTool):
    name = "federal_register"
    description = (
        "Search official U.S. Federal Register documents (rules, proposed "
        "rules, and notices) by keyword. No API key required."
    )

    BASE_URL = "https://www.federalregister.gov/api/v1"
    input_schema: ClassVar[type[BaseModel] | None] = FederalRegisterInput
    MAX_RESULTS: ClassVar[int] = 10
    MAX_ABSTRACT_CHARS: ClassVar[int] = 2_000

    # Fields requested from API
    FIELDS = (
        "title",
        "type",
        "abstract",
        "document_number",
        "html_url",
        "publication_date",
        "effective_on",
        "agencies",
    )

    def run(
        self,
        query: str,
        per_page: int = 5,
        order: str = "newest",
        **_: Any,
    ) -> ToolResult[dict[str, Any]]:
        """Search Federal Register documents.

        Args:
            query: Full-text search term.
            per_page: Number of documents to return (API max 1000).
            order: Sort order — "newest", "oldest", or "relevance".
        """
        try:
            bounded_page = max(1, min(per_page, self.MAX_RESULTS))
            params: dict[str, Any] = {
                "conditions[term]": query,
                "per_page": bounded_page,
                "order": order,
                "fields[]": list(self.FIELDS),
            }
            response = http_get(f"{self.BASE_URL}/documents.json", params=params)
            payload = response.json()

            raw_documents = payload.get("results", [])
            documents = [
                self._parse(item)
                for item in raw_documents[: self.MAX_RESULTS]
            ]
            text_truncated = any(
                isinstance(item.get("abstract"), str)
                and len(item["abstract"]) > self.MAX_ABSTRACT_CHARS
                for item in raw_documents[: self.MAX_RESULTS]
            )
            data = {
                "query": query,
                "total_count": payload.get("count", 0),
                "returned": len(documents),
                "documents": documents,
            }
            return ToolResult(
                success=True,
                data=data,
                source=f"federal_register:{query}",
                metadata={
                    "provider": "federalregister.gov",
                    "total_count": payload.get("count", 0),
                    "returned": len(documents),
                    "order": order,
                    "limit_applied": (
                        per_page != bounded_page
                        or len(raw_documents) > self.MAX_RESULTS
                        or text_truncated
                    ),
                },
            )
        except Exception as exc:
            self.logger.exception("federal_register failed for query=%r", query)
            return ToolResult(
                success=False,
                source=f"federal_register:{query}",
                error=str(exc),
                metadata={"provider": "federalregister.gov"},
            )

    @staticmethod
    def _parse(item: dict[str, Any]) -> dict[str, Any]:
        """Flatten a Federal Register API document into a compact dict."""
        agencies = [a.get("name") for a in (item.get("agencies") or []) if a and a.get("name")]
        return {
            "title": item.get("title"),
            "type": item.get("type"),
            "abstract": FederalRegisterTool.truncate_text(
                item.get("abstract"),
                FederalRegisterTool.MAX_ABSTRACT_CHARS,
            ),
            "document_number": item.get("document_number"),
            "url": item.get("html_url"),
            "publication_date": item.get("publication_date"),
            "effective_on": item.get("effective_on"),
            "agencies": agencies,
        }
