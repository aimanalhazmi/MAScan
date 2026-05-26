"""Tools used ONLY by the Economics agent."""

from mascan.agents.economics.tools.market_data import WebQueryTool, WeeklyStockPricesTool
from mascan.tools.registry import tool_registry

tool_registry.register(WebQueryTool())
tool_registry.register(WeeklyStockPricesTool())

__all__ = ["WebQueryTool", "WeeklyStockPricesTool"]
