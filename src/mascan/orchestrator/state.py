from pydantic import BaseModel, Field
from typing import Annotated
from mascan.contracts.reports import AgentReport


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class GraphState(BaseModel):
    """State carried through the orchestrator graph."""
    user_input: str
    plan: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Planner output: agent_name -> tasks assigned.",
    )
    reports: Annotated[dict[str, AgentReport], merge_dicts] = Field(
        default_factory=dict,
        description="Successful agent reports keyed by agent name.",
    )

    failures: Annotated[dict[str, str], merge_dicts] = Field(
        default_factory=dict,
        description="Agents that errored: agent_name -> error message.",
    )

    final_summary: str = Field("", description="LLM-synthesized final answer.")
    final_markdown: str = Field("", description="Markdown rendering of the final answer.")

    iteration: int = 0
    max_iterations: int = 10

    model_config = {"arbitrary_types_allowed": True}