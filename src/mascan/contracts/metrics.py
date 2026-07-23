"""Typed execution metrics shared by parent and private graphs."""

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class AgentCallMetrics(BaseModel):
    run_count: int = Field(default=0, ge=0)
    token_usage: TokenUsage = Field(
        default_factory=lambda: TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
    )


class ComponentMetrics(BaseModel):
    run_count: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    token_usage: TokenUsage = Field(
        default_factory=lambda: TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
    )
    agents: dict[str, AgentCallMetrics] = Field(default_factory=dict)


def merge_agent_metrics(
    left: dict[str, AgentCallMetrics],
    right: dict[str, AgentCallMetrics],
) -> dict[str, AgentCallMetrics]:
    merged = {name: metric.model_copy(deep=True) for name, metric in left.items()}
    for name, incoming in right.items():
        current = merged.get(name, AgentCallMetrics())
        merged[name] = AgentCallMetrics(
            run_count=current.run_count + incoming.run_count,
            token_usage=TokenUsage(
                input_tokens=(current.token_usage.input_tokens + incoming.token_usage.input_tokens),
                output_tokens=(
                    current.token_usage.output_tokens + incoming.token_usage.output_tokens
                ),
                total_tokens=(current.token_usage.total_tokens + incoming.token_usage.total_tokens),
            ),
        )
    return merged


def merge_component_metrics(
    left: dict[str, ComponentMetrics],
    right: dict[str, ComponentMetrics],
) -> dict[str, ComponentMetrics]:
    merged = {name: metric.model_copy(deep=True) for name, metric in left.items()}
    for name, incoming in right.items():
        current = merged.get(name, ComponentMetrics())
        merged[name] = ComponentMetrics(
            run_count=current.run_count + incoming.run_count,
            duration_seconds=round(
                current.duration_seconds + incoming.duration_seconds,
                6,
            ),
            token_usage=TokenUsage(
                input_tokens=(current.token_usage.input_tokens + incoming.token_usage.input_tokens),
                output_tokens=(
                    current.token_usage.output_tokens + incoming.token_usage.output_tokens
                ),
                total_tokens=(current.token_usage.total_tokens + incoming.token_usage.total_tokens),
            ),
            agents=merge_agent_metrics(current.agents, incoming.agents),
        )
    return merged
