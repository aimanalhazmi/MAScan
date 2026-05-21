"""Tools: how agents fetch live data from EXTERNAL systems (APIs, web, etc.).
"""

from mascan.tools.base import BaseTool
from mascan.tools.registry import ToolRegistry, tool_registry

__all__ = ["BaseTool", "ToolRegistry", "tool_registry"]