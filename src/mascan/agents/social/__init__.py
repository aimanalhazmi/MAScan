"""Social agent — analyzes official social context and public discussion."""

import mascan.agents.social.tools  # noqa: F401  # register agent-specific tools
import mascan.tools.common  # noqa: F401  # register shared tools
from mascan.agents.registry import agent_registry
from mascan.agents.social.agent import SocialAgent

agent_registry.register(SocialAgent())

__all__ = ["SocialAgent"]
