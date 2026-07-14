from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from mascan.contracts.planning import AgentAssignment, InformationRequest
from mascan.contracts.reports import AgentReport


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class RuntimeContext(BaseModel):
    """Runtime metadata shared by graph nodes and optionally passed to agents."""

    current_date: str
    timezone: str

    @classmethod
    def from_system(cls) -> "RuntimeContext":
        """Create date context once at the graph entry boundary."""
        timezone = "Europe/Berlin"
        current_date = datetime.now(ZoneInfo(timezone)).date().isoformat()
        return cls(current_date=current_date, timezone=timezone)


class GraphState(BaseModel):
    """State carried through the orchestrator graph."""
    user_input: str
    # Runtime metadata available throughout a single graph execution.
    runtime_context: RuntimeContext = Field(default_factory=RuntimeContext.from_system)
    plan: dict[str, AgentAssignment] = Field(
        default_factory=dict,
        description="Planner output: agent_name -> assignment.",
    )

    info_request: InformationRequest | None = Field(
        default=None,
        description="If the planner thinks the user question lacks crucial information, it can ask for clarification here instead of returning any assignments. The orchestrator will then ask the user this question and re-run the planner with the user's answer appended to user_input.",
    )

    info_request_counter: int = Field(
        default=0,
        description="Number of times the planner has requested more information. Used to prevent infinite loops in case the planner keeps asking for more info without ever returning a plan.",
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
