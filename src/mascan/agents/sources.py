"""Shared source extraction and rendering for all agents."""

import ast
import json
import re
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit

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
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
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
    for tool_name, result in outputs.items():
        if not result.success:
            continue
        links = links_from_structure(result.data)
        # Some tools (e.g. World Bank) expose links only via metadata.
        for url in result.metadata.get("source_urls") or []:
            if isinstance(url, str) and url.startswith("http"):
                links.append((None, url))
        sources.extend(make_source(label, url, tool_name) for label, url in links)
    return sources


def sources_from_react(result: dict[str, Any]) -> list[Source]:
    """Article-level sources from a ReAct agent's tool-call message history."""
    sources: list[Source] = []
    for msg in result.get("messages", []):
        if getattr(msg, "type", None) != "tool":
            continue
        tool = getattr(msg, "name", None) or "llm_tool"
        sources.extend(
            make_source(label, url, tool) for label, url in links_from_content(msg.content)
        )
    return sources


def dedupe_sources(sources: list[Source]) -> list[Source]:
    """Drop duplicate references, keyed by URL (falling back to name)."""
    seen: dict[str, Source] = {}
    for source in sources:
        key = canonical_source_url(source.url) if source.url else source.name.strip()
        if key not in seen:
            seen[key] = source
    return list(seen.values())


def provided_sources_from_context(context: dict[str, Any] | None) -> list[Source]:
    """Return orchestrator-provided sources that agents may cite directly."""
    values = (context or {}).get("provided_sources")
    if not isinstance(values, list):
        return []

    sources: list[Source] = []
    for value in values:
        try:
            source = value if isinstance(value, Source) else Source.model_validate(value)
        except (TypeError, ValueError):
            continue
        if source.url:
            sources.append(source)
    return dedupe_sources(sources)


def cited_provided_sources(
    findings: str,
    context: dict[str, Any] | None,
) -> list[Source]:
    """Keep only provided sources whose exact links appear in Agent findings."""
    return [
        source
        for source in provided_sources_from_context(context)
        if source.url and f"]({source.url})" in findings
    ]


def canonical_source_url(url: str | None) -> str:
    """Normalize a URL for comparison without discarding meaningful queries."""
    if not url:
        return ""
    cleaned = url.strip()
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return cleaned

    path = parts.path
    if path == "/":
        path = ""
    elif path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def normalize_agent_citations(
    findings: str,
    sources: list[Source],
) -> tuple[str, list[Source]]:
    """Number verified Markdown links and order sources by first citation.

    Links outside the tool-derived source registry are downgraded to plain text.
    When the body contains no verified links, all URL-backed sources are retained
    in their first-collected order as a best-effort fallback.
    """
    deduped = dedupe_sources(sources)
    sources_by_url = {
        canonical_source_url(source.url): source
        for source in deduped
        if source.url and canonical_source_url(source.url)
    }
    cited_urls: list[str] = []
    number_by_url: dict[str, int] = {}

    def replace_link(match: re.Match[str]) -> str:
        label, url = match.groups()
        key = canonical_source_url(url)
        source = sources_by_url.get(key)
        if source is None:
            return label
        if key not in number_by_url:
            number_by_url[key] = len(number_by_url) + 1
            cited_urls.append(key)
        return f"[{number_by_url[key]}]({source.url})"

    normalized_findings = MARKDOWN_LINK_RE.sub(replace_link, findings)
    if cited_urls:
        ordered_sources = [sources_by_url[url] for url in cited_urls]
    else:
        ordered_sources = [source for source in deduped if source.url]
    return normalized_findings, ordered_sources


def cited_tools(
    findings: str,
    sources: list[Source],
) -> list[str]:
    """Return tools whose collected URLs are actually cited in the findings."""
    cited_urls = {canonical_source_url(url) for _, url in MARKDOWN_LINK_RE.findall(findings)}
    tools: list[str] = []
    for source in sources:
        if not source.url or canonical_source_url(source.url) not in cited_urls:
            continue
        tool = source.metadata.get("tool")
        if isinstance(tool, str) and tool not in tools:
            tools.append(tool)
    return tools


def format_source_line(source: Source) -> str:
    if source.url:
        return f"- [{source.name}]({source.url})"
    return f"- {source.name}"


def render_source_lines(sources: list[Source]) -> str:
    if not sources:
        return "- (none)"
    return "\n".join(format_source_line(source) for source in sources)


def render_numbered_source_lines(sources: list[Source]) -> str:
    if not sources:
        return "- (none)"
    return "\n".join(
        f"{number}. [{source.name}]({source.url})"
        for number, source in enumerate(sources, start=1)
        if source.url
    )
