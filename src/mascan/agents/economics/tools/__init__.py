"""Tools used ONLY by the Economics agent."""

from mascan.agents.economics.tools.market_data import WeeklyStockPricesTool
from mascan.tools.registry import tool_registry

tool_registry.register(WeeklyStockPricesTool())

__all__ = ["WeeklyStockPricesTool"]
