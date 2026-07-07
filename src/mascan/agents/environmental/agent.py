from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from mascan.agents import BaseAgent
from mascan.agents.environmental.graph import (
    EnvironmentalAgentState,
    build_environmental_graph,
)
from mascan.agents.environmental.prompts import build_user_prompt
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model


class EnvironmentalAgent(BaseAgent):
    name = "environmental"  # must match config.yaml `name`

    def _run(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> AgentReport:
        self.logger.info(f"Running environmental agent with {len(tasks)} task(s)")

        state = EnvironmentalAgentState(
            tasks=tasks,
            context=context,
            deterministic_outputs=deterministic_outputs or {},
        )
        final_state = self.build_graph().invoke(state)
        report = final_state.get("report") if isinstance(final_state, dict) else None
        if not isinstance(report, AgentReport):
            raise RuntimeError("Environmental graph completed without an AgentReport.")
        return report

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
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": self.config.max_llm_iterations},
        )

        return result, self.extract_final_answer(result), self.extract_used_tools(result)
