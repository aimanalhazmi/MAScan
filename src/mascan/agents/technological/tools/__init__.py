"""Tools used ONLY by the technological agent."""

from mascan.agents.technological.tools.scholar import ScholarSearchTool
from mascan.core.settings import get_settings
from mascan.tools.registry import tool_registry

_settings = get_settings()

tool_registry.register(
    ScholarSearchTool(
        api_key=_settings.semantic_scholar_api_key, api_url=_settings.semantic_scholar_api_url
    )
)

__all__ = ["ScholarSearchTool"]
