"""Economics agent — analyzes the 'E' in PESTEL."""

from mascan.agents.economics.agent import EconomicsAgent
from mascan.agents.registry import agent_registry
from mascan.tools.registry import tool_registry

# Register this agent's private tools BEFORE instantiating the agent
# tool_registry.register(tool())

# Register the agent itself
agent_registry.register(EconomicsAgent())

__all__ = ["EconomicsAgent"]