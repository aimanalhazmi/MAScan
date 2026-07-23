"""Environmental agent — analyzes the 'E' in PESTEL."""

import mascan.agents.environmental.tools  # noqa: F401  # register agent-specific tools
from mascan.agents.environmental.agent import EnvironmentalAgent
from mascan.agents.registry import agent_registry

agent_registry.register(EnvironmentalAgent())

__all__ = ["EnvironmentalAgent"]
