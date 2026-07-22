"""Private LangGraph workflow for EnvironmentalAgent."""

from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from mascan.contracts.metrics import AgentCallMetrics, merge_agent_metrics
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.metrics import measure_agent_call


class EnvironmentalAgentState(BaseModel):
    """State internal to the Environmental agent's private graph."""

    tasks: list[str]
    context: dict[str, Any] | None = None
    deterministic_outputs: dict[str, ToolResult[Any]] = Field(default_factory=dict)
    react_result: dict[str, Any] | None = None
    findings: str = ""
    llm_used_tools: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    agent_metrics: Annotated[dict[str, AgentCallMetrics], merge_agent_metrics] = Field(
        default_factory=dict
    )
    report: AgentReport | None = None

    model_config = {"arbitrary_types_allowed": True}


def build_environmental_graph(agent: Any) -> Any:
    """Build the behavior-preserving private graph for EnvironmentalAgent."""

    def run_react_agent(state: EnvironmentalAgentState) -> dict[str, Any]:
        (react_result, findings, llm_used_tools), agent_metrics = measure_agent_call(
            "analyst",
            lambda: agent.run_react_agent(
                state.tasks,
                context=state.context,
            ),
        )
        return {
            "react_result": react_result,
            "findings": findings,
            "llm_used_tools": llm_used_tools,
            "agent_metrics": agent_metrics,
        }

    def collect_sources(state: EnvironmentalAgentState) -> dict[str, Any]:
        return {
            "sources": agent.collect_sources(
                deterministic_outputs=state.deterministic_outputs,
                react_result=state.react_result,
            )
        }

    def build_report(state: EnvironmentalAgentState) -> dict[str, Any]:
        rendered = agent.render_markdown(
            state.tasks,
            state.findings,
            state.sources,
            state.llm_used_tools,
        )
        return {
            "report": AgentReport(
                agent_name=agent.name,
                tasks=state.tasks,
                findings=state.findings,
                sources=state.sources,
                confidence=0.7,
                rendered_markdown=rendered,
                metadata={
                    "mode": "B — LLM-driven",
                    "deterministic_tools": [],
                    "llm_chosen_tools": state.llm_used_tools,
                },
            )
        }

    graph = StateGraph(EnvironmentalAgentState)
    graph.add_node("run_react_agent", run_react_agent)
    graph.add_node("collect_sources", collect_sources)
    graph.add_node("build_report", build_report)

    graph.add_edge(START, "run_react_agent")
    graph.add_edge("run_react_agent", "collect_sources")
    graph.add_edge("collect_sources", "build_report")
    graph.add_edge("build_report", END)

    return graph.compile()
