"""Cross-agent tools usable by ANY agent.."""

from mascan.tools.common.web_search import WebSearchTool
from mascan.tools import tool_registry

# Register common tools at import time
tool_registry.register(WebSearchTool())

__all__ = ["WebSearchTool"]