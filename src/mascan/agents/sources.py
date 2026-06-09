"""Shared source extraction and rendering for all agents."""

import ast
import json
import re
from typing import Any
from urllib.parse import urlparse

from mascan.contracts.reports import Source
from mascan.contracts.tools import ToolResult

# Dict keys that hold a reference URL, in priority order.
URL_KEYS: tuple[str, ...] = ("url", "api_url", "link", "permalink")
# Dict keys that make a good human-readable label for a reference.
LABEL_KEYS: tuple[str, ...] = (
    "title",
    "headline",
    "indicator_name",
    "name",
    "text",
    "snippet",
)
# Long free-text fields are skipped when recursing so we don't regex-harvest the
# hundreds of incidental links inside a scraped page body.
SKIP_RECURSE_KEYS: frozenset[str] = frozenset(
    {"markdown", "selftext", "snippet", "text", "body", "content", "html"}
)
URL_RE = re.compile(r"https?://[^\s\"'\)\]]+")
MAX_LABEL_CHARS = 160


def label(d: dict[str, Any]) -> str | None:
    for key in LABEL_KEYS:
        value = d.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:MAX_LABEL_CHARS]
    return None


def links_from_structure(obj: Any) -> list[tuple[str | None, str]]:
    """Walk a parsed payload for (label, url) pairs from url-bearing dicts."""
    found: list[tuple[str | None, str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in URL_KEYS:
                url = node.get(key)
                if isinstance(url, str) and url.startswith("http"):
                    found.append((label(node), url))
                    break
            for key, value in node.items():
                if key not in SKIP_RECURSE_KEYS:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(obj)
    return found


def coerce(content: Any) -> Any:
    """Best-effort decode of a ToolMessage payload (JSON, dict-repr, or raw)."""
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(content)
        except (ValueError, SyntaxError, TypeError):
            continue
    return content  # unparseable string — handled by the regex fallback


def links_from_content(content: Any) -> list[tuple[str | None, str]]:
    data = coerce(content)
    if isinstance(data, (dict, list)):
        return links_from_structure(data)
    if isinstance(data, str):
        return [(None, url) for url in URL_RE.findall(data)]
    return []


def make_source(label: str | None, url: str, tool: str) -> Source:
    name = label or urlparse(url).netloc or url
    return Source(name=name, url=url, metadata={"tool": tool})


def sources_from_tool_results(outputs: dict[str, ToolResult[Any]]) -> list[Source]:
    """Article-level sources from deterministic tool outputs."""
    sources: list[Source] = []
    for result in outputs.values():
        if not result.success:
            continue
        links = links_from_structure(result.data)
        # Some tools (e.g. World Bank) expose links only via metadata.
        for url in result.metadata.get("source_urls") or []:
            if isinstance(url, str) and url.startswith("http"):
                links.append((None, url))
        sources.extend(make_source(label, url, result.source) for label, url in links)
    return sources


def sources_from_react(result: dict[str, Any]) -> list[Source]:
    """Article-level sources from a ReAct agent's tool-call message history."""
    sources: list[Source] = []
    for msg in result.get("messages", []):
        if getattr(msg, "type", None) != "tool":
            continue
        tool = getattr(msg, "name", None) or "llm_tool"
        sources.extend(
            make_source(label, url, tool)
            for label, url in links_from_content(msg.content)
        )
    return sources


def dedupe_sources(sources: list[Source]) -> list[Source]:
    """Drop duplicate references, keyed by URL (falling back to name)."""
    seen: dict[str, Source] = {}
    for source in sources:
        key = source.url or source.name
        if key not in seen:
            seen[key] = source
    return list(seen.values())


def format_source_line(source: Source) -> str:
    if source.url:
        return f"- [{source.name}]({source.url})"
    return f"- {source.name}"


def render_source_lines(sources: list[Source]) -> str:
    if not sources:
        return "- (none)"
    return "\n".join(format_source_line(source) for source in sources)
