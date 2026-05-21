"""Reports returned by agents to the orchestrator."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single citation: where a piece of information came from."""

    name: str = Field(..., description="Human-readable source name, e.g. 'FRED:GDP'.")
    url: str | None = Field(None, description="URL if applicable.")
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentReport(BaseModel):
    """Standard output of every agent."""

    agent_name: str = Field(..., description="Identifier of the agent that produced this report.")
    tasks: list[str] = Field(default_factory=list, description="Tasks the agent was asked to do.")
    findings: str = Field(..., description="Plain-text analytical findings.")
    sources: list[Source] = Field(default_factory=list, description="Citations.")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Self-reported confidence [0,1].")
    rendered_markdown: str = Field(..., description="Human-readable markdown rendering.")
    metadata: dict[str, Any] = Field(default_factory=dict)