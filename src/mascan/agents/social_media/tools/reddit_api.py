import os
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool
from mascan.tools.http_client import http_get


class RedditSearchInput(BaseModel):
    query: str = Field(description="Keyword query to search across Reddit.")
    limit: int = Field(10, ge=1, le=25, description="Maximum number of posts to return.")
    sort: str = Field("relevance", description="Search sort: relevance, hot, top, new, or comments.")
    time_filter: str = Field("month", description="Time window: hour, day, week, month, year, or all.")


class RedditSearchTool(BaseTool):
    name = "reddit_search"
    description = (
        "Search Reddit posts for qualitative community discussion, user pain points, "
        "sentiment, and demand signals related to a business question."
    )
    input_schema = RedditSearchInput

    def run(
        self,
        query: str,
        limit: int = 10,
        sort: str = "relevance",
        time_filter: str = "month",
        **_: Any,
    ) -> ToolResult[list[dict[str, Any]]]:
        try:
            user_agent = os.getenv("REDDIT_USER_AGENT", "mascan-social-media-agent/0.1")
            response = http_get(
                "https://www.reddit.com/search.json",
                params={
                    "q": query,
                    "limit": min(limit, 25),
                    "sort": sort,
                    "t": time_filter,
                    "type": "link",
                },
                headers={"User-Agent": user_agent},
            )
            payload = response.json()
            posts = self._format_posts(payload)
            return ToolResult(
                success=True,
                data=posts,
                source="reddit:search",
                metadata={
                    "platform": "reddit",
                    "provider": "reddit_public_json",
                    "query": query,
                    "count": len(posts),
                    "sort": sort,
                    "time_filter": time_filter,
                },
            )
        except Exception as exc:
            self.logger.exception("reddit_search failed for query=%r", query)
            return ToolResult(
                success=False,
                source="reddit:search",
                error=str(exc),
                metadata={
                    "platform": "reddit",
                    "provider": "reddit_public_json",
                    "query": query,
                    "sort": sort,
                    "time_filter": time_filter,
                },
            )

    @staticmethod
    def _format_posts(payload: dict[str, Any]) -> list[dict[str, Any]]:
        children = payload.get("data", {}).get("children", [])
        posts: list[dict[str, Any]] = []
        for child in children:
            data = child.get("data", {})
            permalink = data.get("permalink")
            created_utc = data.get("created_utc")
            created_at = None
            if created_utc is not None:
                created_at = datetime.fromtimestamp(created_utc, tz=UTC).isoformat()

            posts.append(
                {
                    "title": data.get("title"),
                    "subreddit": data.get("subreddit_name_prefixed") or data.get("subreddit"),
                    "score": data.get("score"),
                    "comments": data.get("num_comments"),
                    "url": f"https://www.reddit.com{permalink}" if permalink else data.get("url"),
                    "created_at": created_at,
                    "author": data.get("author"),
                    "snippet": data.get("selftext") or data.get("selftext_html") or data.get("url"),
                }
            )
        return posts
