import os
from typing import Any

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.http_client import http_get


class XSearchInput(BaseModel):
    query: str = Field(description="Keyword query for recent public X posts.")
    max_results: int = Field(10, ge=10, le=100, description="Number of recent posts to return.")


class XSearchTool(BaseTool):
    name = "x_search"
    description = (
        "Search recent public X posts for real-time social discussion, trend signals, "
        "hashtags, and sentiment around a business question."
    )
    input_schema = XSearchInput

    def run(self, query: str, max_results: int = 10, **_: Any) -> ToolResult[list[dict[str, Any]]]:
        token = os.getenv("X_BEARER_TOKEN")
        if not token:
            return ToolResult(
                success=False,
                source="x:recent_search",
                error="X_BEARER_TOKEN is not configured.",
                metadata={
                    "platform": "x",
                    "provider": "x_api_v2",
                    "query": query,
                    "count": 0,
                },
            )

        try:
            response = http_get(
                "https://api.x.com/2/tweets/search/recent",
                params={
                    "query": query,
                    "max_results": max(10, min(max_results, 100)),
                    "tweet.fields": "created_at,author_id,lang,public_metrics",
                },
                headers={"Authorization": f"Bearer {token}"},
            )
            payload = response.json()
            posts = self._format_posts(payload)
            return ToolResult(
                success=True,
                data=posts,
                source="x:recent_search",
                metadata={
                    "platform": "x",
                    "provider": "x_api_v2",
                    "query": query,
                    "count": len(posts),
                },
            )
        except Exception as exc:
            self.logger.exception("x_search failed for query=%r", query)
            return ToolResult(
                success=False,
                source="x:recent_search",
                error=str(exc),
                metadata={
                    "platform": "x",
                    "provider": "x_api_v2",
                    "query": query,
                    "count": 0,
                },
            )

    @staticmethod
    def _format_posts(payload: dict[str, Any]) -> list[dict[str, Any]]:
        posts: list[dict[str, Any]] = []
        for item in payload.get("data", []):
            post_id = item.get("id")
            posts.append(
                {
                    "id": post_id,
                    "text": item.get("text"),
                    "author_id": item.get("author_id"),
                    "created_at": item.get("created_at"),
                    "lang": item.get("lang"),
                    "public_metrics": item.get("public_metrics"),
                    "url": f"https://x.com/i/web/status/{post_id}" if post_id else None,
                }
            )
        return posts
