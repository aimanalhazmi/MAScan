"""Contracts for the planner's output."""

from pydantic import BaseModel, Field


class AgentAssignment(BaseModel):
    """Planner assignment for one domain agent."""

    agent_name: str = Field(description="Name of the selected agent.")
    objective_context: str = Field(
        description="Agent-specific brief preserving the user's intent and constraints.",
    )
    tasks: list[str] = Field(description="Specific sub-tasks assigned to this agent.")
    evidence_documents: list[str] = Field(
        default_factory=list,
        description=(
            "Exact uploaded-document filenames whose retrieved evidence should be "
            "provided to this agent."
        ),
    )
    salient_factors: list[str] = Field(
        default_factory=list,
        description=(
            "3-6 concrete, subject-specific factors this agent must cover for this "
            "dimension (e.g. named regulations, cost drivers, technologies). These are "
            "investigation targets/hypotheses the agent should verify with evidence, "
            "NOT asserted facts about the case."
        ),
    )


class InformationRequest(BaseModel):
    """Planner's request for more information from the user."""

    question: str = Field(
        description="A request for additional information or clarification from the user."
    )

class PlanModel(BaseModel):
    """Structured output the planner LLM is forced to return."""

    assignments: list[AgentAssignment] | InformationRequest = Field(
        description="List of agent-task assignments or a request for more information.",
    )


class IntentCheck(BaseModel):
    """Structured output of the pre-planning intent confirmation step."""

    needs_clarification: bool = Field(
        description="True if the request is too ambiguous to plan confidently.",
    )
    question: str = Field(
        default="",
        description="The clarifying question to ask, when needs_clarification is true.",
    )
