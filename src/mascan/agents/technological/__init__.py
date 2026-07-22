"""Technological agent — analyzes the 'T' in PESTEL."""

import mascan.agents.technological.tools  # noqa: F401  # register agent-specific tools
from mascan.agents.registry import agent_registry
from mascan.agents.technological.agent import TechnologicalAgent

agent_registry.register(TechnologicalAgent())

__all__ = ["TechnologicalAgent"]
