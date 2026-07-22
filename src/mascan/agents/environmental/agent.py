from typing import Any

from langchain.agents import create_agent

from mascan.agents import GraphBackedAgent
from mascan.agents.environmental.graph import (
    EnvironmentalAgentState,
    build_environmental_graph,
)
from mascan.agents.environmental.prompts import build_user_prompt
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model


class EnvironmentalAgent(GraphBackedAgent):
    name = "environmental"  # must match config.yaml `name`

    def build_initial_state(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> EnvironmentalAgentState:
        self.logger.info(f"Running environmental agent with {len(tasks)} task(s)")
        return EnvironmentalAgentState(
            tasks=tasks,
            context=context,
            deterministic_outputs=deterministic_outputs or {},
        )

    def build_graph(self) -> Any:
        return build_environmental_graph(self)

    def run_react_agent(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, list[str]]:
        llm = get_chat_model(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        tools = [tool.as_langchain_tool() for tool in self.optional_tools.values()]
        agent = create_agent(model=llm, tools=tools, system_prompt=self.config.system_prompt)

        user_prompt = build_user_prompt(
            tasks,
            context=context,
        )
        result = self.invoke_react_with_fallback(agent, llm, user_prompt)

        return result, self.extract_final_answer(result), self.extract_used_tools(result)
