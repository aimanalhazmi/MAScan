"""Standard return shape for every tool."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):  # noqa: UP046  # keep explicit Generic[T]; PEP 695 modernization deferred
    """Uniform contract returned by every BaseTool."""

    success: bool
    data: T | None = None
    source: str = Field(..., description="Source identifier, e.g. 'web_search:google'.")
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
