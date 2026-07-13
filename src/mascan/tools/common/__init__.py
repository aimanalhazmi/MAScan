"""Cross-agent tools usable by ANY agent.."""

from mascan.core.settings import get_settings
from mascan.tools.common.rag_search import RagSearchTool
from mascan.tools.common.web_search import WebSearchTool
from mascan.tools.registry import tool_registry

# Register common tools at import time
settings = get_settings()
tool_registry.register(
    WebSearchTool(api_key=settings.firecrawl_api_key, api_url=settings.firecrawl_api_url)
)

if settings.database_url:
    tool_registry.register(RagSearchTool())

__all__ = ["RagSearchTool", "WebSearchTool"]
