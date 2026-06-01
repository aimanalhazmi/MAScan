"""Tools used only by the Social Media agent."""

from mascan.agents.social_media.tools.reddit_api import RedditSearchTool
from mascan.agents.social_media.tools.x_api import XSearchTool
from mascan.tools.registry import tool_registry

tool_registry.register(RedditSearchTool())
tool_registry.register(XSearchTool())

__all__ = ["RedditSearchTool", "XSearchTool"]
