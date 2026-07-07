"""PoliticalAgent — Mode C (mixed)."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from mascan.agents.base import BaseAgent
from mascan.agents.context import render_tool_outputs
from mascan.agents.political.graph import PoliticalAgentState, build_political_graph
from mascan.agents.political.prompts import build_user_prompt
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model


class PoliticalAgent(BaseAgent):
    name = "political"

    def _run(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> AgentReport:
        self.logger.info(f"Running Mode C (mixed) with {len(tasks)} task(s)")

        state = PoliticalAgentState(
            tasks=tasks,
            context=context,
            deterministic_outputs=deterministic_outputs or {},
        )
        final_state = self.build_graph().invoke(state)
        report = final_state.get("report") if isinstance(final_state, dict) else None
        if not isinstance(report, AgentReport):
            raise RuntimeError("Political graph completed without an AgentReport.")
        return report

    def build_graph(self) -> Any:
        return build_political_graph(self)

    def run_react_agent(
        self,
        tasks: list[str],
        deterministic_outputs: dict[str, ToolResult[Any]] | None,
        context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, list[str]]:
        """Run a ReAct agent with the optional tools bound."""
        llm = get_chat_model(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        agent = create_agent(
            model=llm,
            tools=self.get_optional_tools(),
            system_prompt=self.config.system_prompt,
        )

        user_prompt = build_user_prompt(
            tasks,
            render_tool_outputs(deterministic_outputs or {}),
            context=context,
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": self.config.max_llm_iterations},
        )
        return result, self.extract_final_answer(result), self.extract_used_tools(result)
