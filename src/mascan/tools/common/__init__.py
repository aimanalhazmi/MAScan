"""Cross-agent tools usable by ANY agent.."""

from mascan.tools.common.web_search import WebSearchTool
from mascan.tools.registry import tool_registry
from mascan.core.settings import get_settings

# Register common tools at import time
_settings = get_settings()
tool_registry.register(
    WebSearchTool(api_key=_settings.firecrawl_api_key, api_url=_settings.firecrawl_api_url)
)

__all__ = ["WebSearchTool"]