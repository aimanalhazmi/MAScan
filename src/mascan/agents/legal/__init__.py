"""Legal agent — analyzes the 'L' in PESTEL."""

import mascan.agents.legal.tools  # noqa: F401  # register agent-specific tools
import mascan.tools.common  # noqa: F401  # register shared tools
from mascan.agents.legal.agent import LegalAgent
from mascan.agents.registry import agent_registry

# Register the agent itself
agent_registry.register(LegalAgent())

__all__ = ["LegalAgent"]
