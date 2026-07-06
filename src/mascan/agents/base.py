"""BaseAgent — the contract every agent must implement."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from mascan.agents.config import AgentConfig
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model
from mascan.core.logging import get_logger
from mascan.tools.base import BaseTool
from mascan.tools.registry import tool_registry

from langchain_core.language_models import BaseChatModel


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
        # self.tools: dict[str, BaseTool] = {**self.always_call_tools, **self.optional_tools}
        self.logger = get_logger(f"agents.{self.name}")

    @classmethod
    def load_default_config(cls) -> AgentConfig:
        """Load this agent's `config.yaml` from its own folder."""
        import inspect
        from pathlib import Path

        module_file = inspect.getfile(cls)
        config_path = Path(module_file).parent / "config.yaml"
        return AgentConfig.from_yaml(config_path)

    def llm_with_tools(self) -> BaseChatModel:
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
        return self._run(tasks, context=context, deterministic_outputs=deterministic_outputs)
    
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
    
    def get_optional_tools(self) -> list:
        """Return LangChain-wrapped tools the LLM is allowed to call."""
        return [
            tool.as_langchain_tool()
            for name, tool in self.optional_tools.items()
            if name in self.config.optional_tools
        ]

    @abstractmethod
    def _run(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None
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

    def __repr__(self) -> str:
        return f"<Agent name={self.name!r} tools={self.config.tools}>"