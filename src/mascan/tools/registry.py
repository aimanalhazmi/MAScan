"""ToolRegistry — central place to look up tools by name."""

from mascan.core.exceptions import RegistryError
from mascan.tools.base import BaseTool


class ToolRegistry:
    """Maps tool names to tool instances.

    Use the module-level `tool_registry` singleton. Register tools at
    import time.
    """

    def __init__(self) -> None:
        self.tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Raises if the name is already taken."""
        if tool.name in self.tools:
            raise RegistryError(f"Tool {tool.name!r} is already registered.")
        self.tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Get a tool by name. Raises if not found."""
        if name not in self.tools:
            raise RegistryError(
                f"Tool {name!r} not found. Registered: {sorted(self.tools.keys())}"
            )
        return self.tools[name]

    def get_many(self, names: list[str]) -> dict[str, BaseTool]:
        """Get multiple tools by name."""
        return {name: self.get(name) for name in names}

    def all_names(self) -> list[str]:
        return sorted(self.tools.keys())

    def clear(self) -> None:
        """Reset the registry. Mainly useful in tests."""
        self.tools.clear()


tool_registry = ToolRegistry()
