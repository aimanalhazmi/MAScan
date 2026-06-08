"""Tools used only by the Social agent."""

from mascan.agents.social.tools.reddit_api import RedditSearchTool
from mascan.agents.social.tools.world_bank import WorldBankSocialIndicatorsTool
from mascan.agents.social.tools.x_api import XSearchTool
from mascan.tools.registry import tool_registry

tool_registry.register(RedditSearchTool())
tool_registry.register(WorldBankSocialIndicatorsTool())
tool_registry.register(XSearchTool())

__all__ = ["RedditSearchTool", "WorldBankSocialIndicatorsTool", "XSearchTool"]
