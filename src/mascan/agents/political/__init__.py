"""Political agent — analyzes the 'P' in PESTEL."""

import mascan.agents.political.tools  # noqa: F401  # register agent-specific tools
from mascan.agents.political.agent import PoliticalAgent
from mascan.agents.registry import agent_registry

agent_registry.register(PoliticalAgent())

__all__ = ["PoliticalAgent"]
