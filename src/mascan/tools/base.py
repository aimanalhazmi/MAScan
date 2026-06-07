"""BaseTool — the contract every tool must implement."""

import functools
import inspect
from abc import ABC, abstractmethod
from typing import  Any, ClassVar

from mascan.contracts.tools import ToolResult
from mascan.core.logging import get_logger

from pydantic import BaseModel
from langchain_core.tools import StructuredTool

class BaseTool(ABC):
    """Abstract base class for all tools.

    A tool fetches data from outside the agent's own reasoning — typically
    an external API, a web search, or a computational utility.

    Subclasses MUST:
      - Set the class attribute `name` (unique, snake_case identifier).
      - Set the class attribute `description` (one-line, what it does).
      - Implement `run(**kwargs) -> ToolResult`.

    Subclasses MUST NOT:
      - Raise on expected errors. Return ToolResult(success=False, ...) instead.
      - Read API keys directly — read them from settings or the constructor.

    Two ways to use a tool:
      1. Code-driven (default): `tool.run(query="...")` — you decide when.
      2. LLM-driven: `tool.as_langchain_tool()` — bind to an LLM that chooses.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    # Optional: override in subclasses to enable typed I/O.
    input_schema: ClassVar[type[BaseModel] | None] = None
    output_schema: ClassVar[type[BaseModel] | None] = None

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set class attribute `name`.")
        if not self.description:
            raise ValueError(f"{type(self).__name__} must set class attribute `description`.")
        self.logger = get_logger(f"tools.{self.name}")

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Kwargs are tool-specific."""
        ...

    def as_langchain_tool(self) -> StructuredTool:
        """Expose this tool as a LangChain StructuredTool for LLM tool-calling.

        Use this when you want the LLM to decide *whether* and *with what
        arguments* to call the tool, instead of calling `run()` directly in
        code. Bind the result to a chat model with `llm.bind_tools([...])`
        or pass it to `create_agent(...)`.

        Returns:
            A LangChain StructuredTool whose name and description come from
            this tool's class attributes.
        """


        @functools.wraps(self.run)
        def invoke(**call_kwargs: Any) -> Any:
            result = self.run(**call_kwargs)
            if not result.success:
                return f"Tool {self.name!r} failed: {result.error}"
            if isinstance(result.data, BaseModel):
                return result.data.model_dump(mode="json")
            return result.data

        invoke.__signature__ = inspect.signature(self.run)

        kwargs: dict[str, Any] = {
            "func": invoke,
            "name": self.name,
            "description": self.description,
        }
        if self.input_schema is not None:
            kwargs["args_schema"] = self.input_schema

        return StructuredTool.from_function(**kwargs)

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"