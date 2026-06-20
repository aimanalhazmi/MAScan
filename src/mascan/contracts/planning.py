"""Planning contracts shared between orchestrator and report models."""

from pydantic import BaseModel, Field


class AgentAssignment(BaseModel):
    """Planner assignment for one domain agent."""

    agent_name: str = Field(description="Name of the selected agent.")
    objective_context: str = Field(
        description="Agent-specific brief preserving the user's intent and constraints.",
    )
    tasks: list[str] = Field(description="Specific sub-tasks assigned to this agent.")

class InformationRequest(BaseModel):
    """Planner's request for more information from the user."""

    question: str = Field(description="A request for additional information or clarification from the user.")
