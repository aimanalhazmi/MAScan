from typing import Any

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.core.settings import get_settings
from mascan.tools.base import BaseTool


class XSearchInput(BaseModel):
    query: str = Field(description="Keyword query for recent public X posts.")
    max_results: int = Field(10, ge=1, le=100, description="Number of recent posts to return.")


class XSearchTool(BaseTool):
    name = "x_search"
    description = (
        "Search recent public X posts via twitter-cli using cached auth tokens for "
        "qualitative public-discussion signals. Requires TWITTER_AUTH_TOKEN and TWITTER_CT0."
    )
    input_schema = XSearchInput

    def run(self, query: str, max_results: int = 10, **_: Any) -> ToolResult[list[dict[str, Any]]]:
        settings = get_settings()
        auth_token = settings.twitter_auth_token
        ct0 = settings.twitter_ct0
        if not auth_token or not ct0:
            # Token-based auth only — never extract from the browser, so this stays
            # non-interactive (no macOS Keychain password prompt mid-run).
            return self._failure(
                query,
                "X not authenticated. Set TWITTER_AUTH_TOKEN and TWITTER_CT0 in your .env.",
            )

        try:
            from twitter_cli.client import TwitterClient
            from twitter_cli.serialization import tweets_to_data
        except ImportError as exc:
            return self._failure(query, f"twitter-cli is not installed: {exc}")

        try:
            client = TwitterClient(auth_token, ct0)
            tweets = client.fetch_search(query, count=max(1, min(max_results, 100)), product="Top")
            posts = self._format_posts(tweets_to_data(tweets))
            return ToolResult(
                success=True,
                data=posts,
                source="x:twitter_cli_search",
                metadata={
                    "platform": "x",
                    "provider": "twitter-cli",
                    "query": query,
                    "count": len(posts),
                },
            )
        except Exception as exc:
            self.logger.exception("x_search failed for query=%r", query)
            return self._failure(query, str(exc))

    @staticmethod
    def _format_posts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        posts: list[dict[str, Any]] = []
        for item in items:
            post_id = item.get("id")
            author = item.get("author") if isinstance(item.get("author"), dict) else {}
            screen_name = author.get("screenName")
            if screen_name and post_id:
                url = f"https://x.com/{screen_name}/status/{post_id}"
            elif post_id:
                url = f"https://x.com/i/web/status/{post_id}"
            else:
                url = None
            posts.append(
                {
                    "id": post_id,
                    "text": item.get("text"),
                    "author": screen_name,
                    "created_at": item.get("createdAtISO") or item.get("createdAt"),
                    "url": url,
                    "metrics": item.get("metrics"),
                    "raw": item,
                }
            )
        return posts

    @staticmethod
    def _failure(query: str, error: str) -> ToolResult[list[dict[str, Any]]]:
        return ToolResult(
            success=False,
            source="x:twitter_cli_search",
            error=error,
            metadata={
                "platform": "x",
                "provider": "twitter-cli",
                "query": query,
                "count": 0,
            },
        )
