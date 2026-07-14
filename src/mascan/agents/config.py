"""AgentConfig — Pydantic schema + YAML loader used by every agent."""
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from mascan.core.exceptions import ConfigError


class AgentConfig(BaseModel):
    """Schema every agent's config.yaml must conform to."""
    name: str = Field(..., description="Unique agent identifier (snake_case).")
    model: str = Field("gpt-4o-mini", description="LLM model name.")
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, gt=0)
    system_prompt: str = Field(..., description="System prompt for the agent.")
    always_call_tools: list[str] = Field(default_factory=list, description="Tool names this agent always calls.")
    optional_tools: list[str] = Field(default_factory=list, description="Tool names this agent may call.")
    max_llm_iterations: int = Field(25, gt=0, description="Maximum number of LLM iterations.")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific options (e.g. feature toggles) read by that agent.",
    )

    @classmethod
    def from_yaml(cls, path: Path | str):
        """Load and validate an agent config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Agent config not found: {path}")
        try:
            with path.open() as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Agent config root must be a mapping: {path}")
        return cls(**data)
