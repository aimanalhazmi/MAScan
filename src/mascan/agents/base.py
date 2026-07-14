"""BaseAgent — the contract every agent must implement."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from mascan.agents.config import AgentConfig
from mascan.agents.sources import (
    cited_tools,
    dedupe_sources,
    normalize_agent_citations,
    render_numbered_source_lines,
    sources_from_react,
    sources_from_tool_results,
)
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger
from mascan.tools.base import BaseTool
from mascan.tools.registry import tool_registry


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Subclasses MUST:
      - Set class attribute `name` (matches `name` in their config.yaml).
      - Implement `run(tasks, context) -> AgentReport`.
      - Place their `config.yaml` next to `agent.py` in the same folder.

    Subclasses get for free:
      - `self.config` — typed AgentConfig loaded from their config.yaml.
      - `self.tools` — dict of tool name -> BaseTool, resolved from registry.
      - `self.logger` — namespaced logger.

    Two ways to use tools inside `run()`:
      1. Code-driven (default): `self.tools["name"].run(...)` directly.
      2. LLM-driven: `self.llm_with_tools()` returns a chat model with
         the agent's tools bound. The LLM decides which to call.
    Mix freely — call some tools directly, let the LLM pick among others.
    """

    name: ClassVar[str] = ""

    def __init__(
        self,
        config: AgentConfig | None = None,
    ) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set class attribute `name`.")

        self.config = config if config is not None else self.load_default_config()
        if self.config.name != self.name:
            raise ValueError(
                f"Config name {self.config.name!r} does not match agent class name {self.name!r}."
            )

        self.always_call_tools: dict[str, BaseTool] = tool_registry.get_many(self.config.always_call_tools)
        self.optional_tools: dict[str, BaseTool] = tool_registry.get_many(self.config.optional_tools)
        self.tools: dict[str, BaseTool] = {**self.always_call_tools, **self.optional_tools}
        self.logger = get_logger(f"agents.{self.name}")

    @abstractmethod
    def _run(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> AgentReport:
        """Execute the agent's analysis.

        Args:
            tasks: Specific subtasks the orchestrator has assigned.
            context: Optional shared state from the orchestrator.
            deterministic_outputs: Results from always-call tools.

        Returns:
            AgentReport with structured fields AND rendered markdown.
        """
        ...

    @classmethod
    def load_default_config(cls) -> AgentConfig:
        """Load this agent's `config.yaml` from its own folder."""
        import inspect
        from pathlib import Path

        module_file = inspect.getfile(cls)
        config_path = Path(module_file).parent / "config.yaml"
        return cast(AgentConfig, AgentConfig.from_yaml(config_path))

    @staticmethod
    def extract_final_answer(result: dict[str, Any]) -> str:
        """Last message in the ReAct result is the LLM's final answer."""
        messages = result.get("messages", [])
        if not messages:
            return "(no response)"
        return str(messages[-1].content)

    @staticmethod
    def extract_used_tools(result: dict[str, Any]) -> list[str]:
        """Walk the message history and collect every tool the LLM invoked."""
        used: list[str] = []
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name and name not in used:
                    used.append(name)
        return used

    def invoke_react_with_fallback(
        self,
        agent: Any,
        llm: BaseChatModel,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Run ReAct and force a final answer if its tool loop hits the step cap."""
        latest: dict[str, Any] | None = None
        try:
            for update in agent.stream(
                {"messages": [HumanMessage(content=user_prompt)]},
                config={"recursion_limit": self.config.max_llm_iterations},
                stream_mode="values",
            ):
                latest = update
            return latest or {"messages": []}
        except GraphRecursionError:
            messages = list((latest or {}).get("messages") or [])
            self.logger.warning(
                "ReAct reached recursion limit %s; forcing a tool-free final report.",
                self.config.max_llm_iterations,
            )
            final = llm.invoke(
                [
                    SystemMessage(content=self.config.system_prompt),
                    *messages,
                    HumanMessage(
                        content=(
                            "The evidence-collection budget is exhausted. Do not call any "
                            "more tools. Write the final concise analysis now using only "
                            "the evidence already present in this conversation. Preserve "
                            "the requested inline URL citations and explicitly acknowledge "
                            "any evidence gaps."
                        )
                    ),
                ]
            )
            return {"messages": [*messages, final]}

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        """Wraps the subclass's `_run()` method with execution of mandatory tool calls.

        Args:
            tasks: Specific subtasks the orchestrator has assigned.
            context: Optional shared state from the orchestrator.

        Returns:
            AgentReport with structured fields AND rendered markdown.
        """

        # Execute always-call tools first, if any.
        deterministic_outputs = self.gather_deterministic(tasks)
        report = self._run(
            tasks,
            context=context,
            deterministic_outputs=deterministic_outputs,
        )
        cited_findings, ordered_sources = normalize_agent_citations(
            report.findings,
            report.sources,
        )
        called_tools = list(report.metadata.get("llm_chosen_tools") or [])
        referenced_tools = cited_tools(cited_findings, ordered_sources)
        cited_llm_tools = [tool for tool in referenced_tools if tool in called_tools]
        default_display_tools = list(report.metadata.get("default_display_tools") or [])
        display_tools = list(dict.fromkeys([*default_display_tools, *cited_llm_tools]))
        rendered = self.render_markdown(
            report.tasks,
            cited_findings,
            ordered_sources,
            display_tools,
        )
        return report.model_copy(
            update={
                "findings": cited_findings,
                "sources": ordered_sources,
                "rendered_markdown": rendered,
                "metadata": {
                    **report.metadata,
                    "llm_called_tools": called_tools,
                    "llm_chosen_tools": display_tools,
                    "cited_tools": referenced_tools,
                },
            }
        )

    def llm_with_tools(self) -> Any:
        """Return an LLM bound to this agent's tools for LLM-driven calling.

        Usage inside `run()`:
            llm = self.llm_with_tools()
            response = llm.invoke([...])
            # If response.tool_calls is non-empty, the LLM wants tool calls.
            # Use LangGraph's create_react_agent for an automatic loop.

        For code-driven tool use, do NOT call this — just use
        `self.tools["name"].run(...)` directly.
        """
        llm = get_chat_model(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        lc_tools = [t.as_langchain_tool() for t in self.optional_tools.values()]
        return llm.bind_tools(lc_tools)

    def gather_deterministic(self, tasks: list[str]) -> dict[str, ToolResult[Any]]:
        """Call the always-call tools regardless of the question."""
        query = " ; ".join(tasks)
        outputs: dict[str, ToolResult[Any]] = {}
        for tool_name in self.always_call_tools:
            if tool_name in self.tools:
                outputs[tool_name] = self.tools[tool_name].run(query=query)
            else:
                self.logger.warning(f"Always-call tool {tool_name!r} not available; skipping.")
        return outputs

    def get_optional_tools(self) -> list[Any]:
        """Return LangChain-wrapped tools the LLM is allowed to call."""
        return [
            tool.as_langchain_tool()
            for name, tool in self.optional_tools.items()
            if name in self.config.optional_tools
        ]

    def collect_sources(
        self,
        react_result: dict[str, ToolResult[Any]] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> list[Source]:
        """Real reference links harvested from the agent's tool calls."""
        sources: list[Source] = []
        if deterministic_outputs:
            sources.extend(sources_from_tool_results(deterministic_outputs))
        if react_result:
            sources.extend(sources_from_react(react_result))
        return dedupe_sources(sources)

    def render_markdown(
        self,
        tasks: list[str],
        findings: str,
        sources: list[Source],
        llm_used_tools: list[str],
    ) -> str:
        task_lines = "\n".join(f"- {t}" for t in tasks)
        src_lines = render_numbered_source_lines(sources)
        tool_block = ""
        if llm_used_tools:
            llm_lines = "\n".join(f"- {tool}" for tool in llm_used_tools)
            tool_block = f"**Tools the LLM chose to call:**\n{llm_lines}\n\n"
        return (
            f"## {self.name.title()} Analysis\n\n"
            f"**Tasks:**\n{task_lines}\n\n"
            f"**Findings:**\n\n{findings}\n\n"
            f"{tool_block}"
            f"**Sources:**\n{src_lines}\n"
        )

    def __repr__(self) -> str:
        return f"<Agent name={self.name!r} tools={list(self.tools)}>"


class GraphBackedAgent(BaseAgent):
    """Base class for agents whose `_run()` delegates to a private graph."""

    def _run(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> AgentReport:
        state = self.build_initial_state(
            tasks,
            context=context,
            deterministic_outputs=deterministic_outputs,
        )
        final_state = self.build_graph().invoke(state)
        return self.extract_report(final_state)

    @abstractmethod
    def build_initial_state(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> Any:
        """Build the initial state for this agent's private graph."""
        ...

    @abstractmethod
    def build_graph(self) -> Any:
        """Build this agent's private graph."""
        ...

    def extract_report(self, final_state: Any) -> AgentReport:
        """Extract the standard agent report from a completed graph state."""
        report = final_state.get("report") if isinstance(final_state, dict) else None
        if not isinstance(report, AgentReport):
            raise RuntimeError(
                f"{type(self).__name__} graph completed without an AgentReport."
            )
        return report
