"""Social Media agent — analyzes public social discussion and sentiment."""

import mascan.agents.social_media.tools  # noqa: F401  # register agent-specific tools
import mascan.tools.common  # noqa: F401  # register shared tools
from mascan.agents.registry import agent_registry
from mascan.agents.social_media.agent import SocialMediaAgent

agent_registry.register(SocialMediaAgent())

__all__ = ["SocialMediaAgent"]
