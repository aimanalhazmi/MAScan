"""Political-agent tools."""

from mascan.agents.political.tools.news_api import NewsDataSearchTool
from mascan.tools.registry import tool_registry

tool_registry.register(NewsDataSearchTool())

__all__ = ["NewsDataSearchTool"]
