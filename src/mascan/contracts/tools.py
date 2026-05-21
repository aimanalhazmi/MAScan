"""Standard return shape for every tool."""

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Uniform contract returned by every BaseTool."""
    success: bool
    data: Any = None
    source: str = Field(..., description="Source identifier, e.g. 'web_search:google'.")
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)