"""AgentRegistry — central registration and lookup for agents."""

from mascan.agents.base import BaseAgent
from mascan.core.exceptions import RegistryError


class AgentRegistry:
    """Maps agent names to agent instances."""

    def __init__(self) -> None:
        self.agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if agent.name in self.agents:
            raise RegistryError(f"Agent {agent.name!r} is already registered.")
        self.agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self.agents:
            raise RegistryError(
                f"Agent {name!r} not found. Registered: {sorted(self.agents.keys())}"
            )
        return self.agents[name]

    def all(self) -> list[BaseAgent]:
        return list(self.agents.values())

    def all_names(self) -> list[str]:
        return sorted(self.agents.keys())

    def clear(self) -> None:
        self.agents.clear()


agent_registry = AgentRegistry()