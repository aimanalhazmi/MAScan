from typing import Any, ClassVar

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool


class RedditSearchInput(BaseModel):
    query: str = Field(description="Keyword query to search across Reddit.")
    limit: int = Field(10, description="Requested posts; values are clamped to 1–10.")
    sort: str = Field("relevance", description="Search sort: relevance, hot, top, new, or comments.")
    time_filter: str = Field("month", description="Time window: hour, day, week, month, year, or all.")


class RedditSearchTool(BaseTool):
    name = "reddit_search"
    description = (
        "Search Reddit posts via the local rdt-cli session for qualitative community "
        "discussion, user pain points, sentiment, and demand signals."
    )
    input_schema = RedditSearchInput
    MAX_RESULTS: ClassVar[int] = 10

    def run(
        self,
        query: str,
        limit: int = 10,
        sort: str = "relevance",
        time_filter: str = "month",
        **_: Any,
    ) -> ToolResult[list[dict[str, Any]]]:
        try:
            from rdt_cli.auth import load_credential
            from rdt_cli.client import RedditClient
            from rdt_cli.parser import parse_listing
        except ImportError as exc:
            return self._failure(query, sort, time_filter, f"rdt-cli is not installed: {exc}")

        # Saved credential only — never extract from the browser, so this stays
        # non-interactive (no macOS Keychain password prompt mid-run).
        credential = load_credential()
        if credential is None:
            return self._failure(
                query,
                sort,
                time_filter,
                "Reddit not authenticated. Run `rdt login` once to cache a session.",
            )

        try:
            bounded_limit = max(1, min(limit, self.MAX_RESULTS))
            with RedditClient(credential) as client:
                payload = client.search(
                    query,
                    sort=sort,
                    time_filter=time_filter,
                    limit=bounded_limit,
                )
            items = parse_listing(payload).items
            selected = items[: self.MAX_RESULTS]
            posts = self._format_posts(selected)
            text_truncated = any(
                isinstance(post.get("snippet"), str)
                and post["snippet"].endswith(" […]")
                for post in posts
            )
            return ToolResult(
                success=True,
                data=posts,
                source="reddit:rdt_cli_search",
                metadata={
                    "platform": "reddit",
                    "provider": "rdt-cli",
                    "query": query,
                    "count": len(posts),
                    "sort": sort,
                    "time_filter": time_filter,
                    "limit_applied": (
                        limit != bounded_limit
                        or len(items) > self.MAX_RESULTS
                        or text_truncated
                    ),
                },
            )
        except Exception as exc:
            self.logger.exception("reddit_search failed for query=%r", query)
            return self._failure(query, sort, time_filter, str(exc))

    # Post bodies can be very long; cap them so a batch of posts cannot overflow
    # the model context window once they accumulate in the ReAct message history.
    MAX_SNIPPET_CHARS: ClassVar[int] = 1000

    @classmethod
    def _format_posts(cls, items: list[Any]) -> list[dict[str, Any]]:
        posts: list[dict[str, Any]] = []
        for post in items:
            data = post.to_dict()
            permalink = data.get("permalink") or ""
            url = data.get("url") or (f"https://reddit.com{permalink}" if permalink else None)
            posts.append(
                {
                    "id": data.get("id"),
                    "title": data.get("title"),
                    "subreddit": data.get("subreddit"),
                    "score": data.get("score"),
                    "comments": data.get("num_comments"),
                    "url": url,
                    "created_at": data.get("created_utc"),
                    "author": data.get("author"),
                    "snippet": cls._truncate(data.get("selftext")),
                }
            )
        return posts

    @classmethod
    def _truncate(cls, text: Any) -> Any:
        return cls.truncate_text(text, cls.MAX_SNIPPET_CHARS)

    @staticmethod
    def _failure(
        query: str,
        sort: str,
        time_filter: str,
        error: str,
    ) -> ToolResult[list[dict[str, Any]]]:
        return ToolResult(
            success=False,
            source="reddit:rdt_cli_search",
            error=error,
            metadata={
                "platform": "reddit",
                "provider": "rdt-cli",
                "query": query,
                "count": 0,
                "sort": sort,
                "time_filter": time_filter,
            },
        )
