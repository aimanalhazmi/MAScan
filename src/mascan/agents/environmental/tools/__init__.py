"""Tools used ONLY by the Environmental agent."""

from mascan.agents.environmental.tools.world_bank import (
    WorldBankEnvironmentalIndicatorsTool,
)
from mascan.tools.registry import tool_registry

tool_registry.register(WorldBankEnvironmentalIndicatorsTool())

__all__ = ["WorldBankEnvironmentalIndicatorsTool"]
