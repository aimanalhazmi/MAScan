import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mascan.contracts.tools import ToolResult
from mascan.tools.base import BaseTool


class XSearchInput(BaseModel):
    query: str = Field(description="Keyword query for recent public X posts.")
    max_results: int = Field(10, ge=1, le=100, description="Number of recent posts to return.")


class XSearchTool(BaseTool):
    name = "x_search"
    description = (
        "Search recent public X posts using the local twitter-cli browser-cookie session. "
        "Requires twitter-cli to be installed and access to logged-in browser cookies."
    )
    input_schema = XSearchInput

    def run(self, query: str, max_results: int = 10, **_: Any) -> ToolResult[list[dict[str, Any]]]:
        command = self._find_command(os.getenv("TWITTER_CLI_COMMAND", "twitter"))
        if command is None:
            return self._failure(
                query=query,
                error="twitter-cli is not installed or not on PATH.",
            )

        try:
            completed = subprocess.run(
                [
                    command,
                    "search",
                    query,
                    "--max",
                    str(max(1, min(max_results, 100))),
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
                source="x:twitter_cli_search",
                metadata={
                    "platform": "x",
                    "provider": "twitter-cli",
                    "query": query,
                    "count": len(posts),
                    "command": Path(command).name,
                },
            )
        except subprocess.CalledProcessError as exc:
            self.logger.exception("twitter-cli search failed for query=%r", query)
            return self._failure(query=query, error=(exc.stderr or exc.stdout or str(exc)).strip())
        except Exception as exc:
            self.logger.exception("x_search failed for query=%r", query)
            return self._failure(query=query, error=str(exc))

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
            for key in ("data", "tweets", "results", "items"):
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
            post_id = item.get("id") or item.get("tweet_id") or item.get("rest_id")
            text = item.get("text") or item.get("full_text") or item.get("content")
            username = item.get("username") or item.get("user") or item.get("screen_name")
            posts.append(
                {
                    "id": post_id,
                    "text": text,
                    "author": username,
                    "created_at": item.get("created_at") or item.get("date"),
                    "url": item.get("url")
                    or (f"https://x.com/i/web/status/{post_id}" if post_id else None),
                    "metrics": item.get("metrics") or item.get("public_metrics"),
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
