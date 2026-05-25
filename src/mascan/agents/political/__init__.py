"""Political agent — analyzes the 'P' in PESTEL."""

import mascan.tools.common  # noqa: F401  # register shared tools
from mascan.agents.political.agent import PoliticalAgent
from mascan.agents.registry import agent_registry

# Register the agent itself
agent_registry.register(PoliticalAgent())

__all__ = ["PoliticalAgent"]
