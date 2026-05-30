"""Cross-agent tools usable by ANY agent.."""

from mascan.tools.common.web_search import WebSearchTool
from mascan.tools.registry import tool_registry
from mascan.core.settings import get_settings

# Register common tools at import time
tool_registry.register(WebSearchTool(api_key=get_settings().firecrawl_api_key))

__all__ = ["WebSearchTool"]