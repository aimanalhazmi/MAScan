"""Technological agent — analyzes the 'T' in PESTEL."""

import mascan.agents.technological  # noqa: F401  # register agent-specific tools
from mascan.agents.technological.agent import TechnologicalAgent
from mascan.agents.registry import agent_registry

agent_registry.register(TechnologicalAgent())

__all__ = ["TechnologicalAgent"]