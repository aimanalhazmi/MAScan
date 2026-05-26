"""Economics agent — analyzes the 'E' in PESTEL."""

import mascan.agents.economics.tools  # noqa: F401  # register agent-specific tools
from mascan.agents.economics.agent import EconomicsAgent
from mascan.agents.registry import agent_registry

# Register the agent itself
agent_registry.register(EconomicsAgent())

__all__ = ["EconomicsAgent"]
