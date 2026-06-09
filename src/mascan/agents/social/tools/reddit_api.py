import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool


class RedditSearchInput(BaseModel):
    query: str = Field(description="Keyword query to search across Reddit.")
    limit: int = Field(10, ge=1, le=25, description="Maximum number of posts to return.")
    sort: str = Field("relevance", description="Search sort: relevance, hot, top, new, or comments.")
    time_filter: str = Field("month", description="Time window: hour, day, week, month, year, or all.")


class RedditSearchTool(BaseTool):
    name = "reddit_search"
    description = (
        "Search Reddit posts using the local rdt-cli browser-cookie session for qualitative "
        "community discussion, user pain points, sentiment, and demand signals."
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
        command = self._find_command(os.getenv("REDDIT_CLI_COMMAND", "rdt"))
        if command is None:
            return self._failure(
                query=query,
                sort=sort,
                time_filter=time_filter,
                error="rdt-cli is not installed or not on PATH.",
            )

        try:
            completed = subprocess.run(
                [
                    command,
                    "search",
                    query,
                    "--limit",
                    str(min(limit, 25)),
                    "--sort",
                    sort,
                    "--time",
                    time_filter,
                    "--compact",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            payload = json.loads(completed.stdout)
            posts = self._format_posts(payload)
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
                    "command": Path(command).name,
                },
            )
        except subprocess.CalledProcessError as exc:
            self.logger.exception("rdt-cli search failed for query=%r", query)
            return self._failure(
                query=query,
                sort=sort,
                time_filter=time_filter,
                error=(exc.stderr or exc.stdout or str(exc)).strip(),
            )
        except Exception as exc:
            self.logger.exception("reddit_search failed for query=%r", query)
            return self._failure(
                query=query,
                sort=sort,
                time_filter=time_filter,
                error=str(exc),
            )

    @staticmethod
    def _find_command(command: str) -> str | None:
        found = shutil.which(command)
        if found:
            return found

        local_bin = Path.home() / ".local" / "bin" / command
        if local_bin.exists():
            return str(local_bin)
        return None

    @staticmethod
    def _extract_items(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "posts", "results", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []

    @classmethod
    def _format_posts(cls, payload: Any) -> list[dict[str, Any]]:
        posts: list[dict[str, Any]] = []
        for item in cls._extract_items(payload):
            if not isinstance(item, dict):
                continue
            post_id = item.get("id") or item.get("name")
            subreddit = item.get("subreddit") or item.get("subreddit_name_prefixed")
            posts.append(
                {
                    "id": post_id,
                    "title": item.get("title"),
                    "subreddit": subreddit,
                    "score": item.get("score") or item.get("ups"),
                    "comments": item.get("comments") or item.get("num_comments"),
                    "url": item.get("url") or item.get("permalink"),
                    "created_at": item.get("created_at") or item.get("created_utc"),
                    "author": item.get("author"),
                    "snippet": item.get("snippet") or item.get("selftext") or item.get("text"),
                    "raw": item,
                }
            )
        return posts

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
