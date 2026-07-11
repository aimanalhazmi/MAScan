"""LegalAgent — Mode C (mixed).

Pattern:
  1. Always call certain tools deterministically (core data we always need).
  2. Pass the results as context to a ReAct agent that decides whether to
     also call optional tools, then writes the final answer.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from mascan.agents.base import GraphBackedAgent
from mascan.agents.legal.graph import LegalAgentState, build_legal_graph
from mascan.agents.legal.prompts import build_user_prompt, render_tool_outputs
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model


class LegalAgent(GraphBackedAgent):
    name = "legal"  # must match config.yaml `name`

    def build_initial_state(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> LegalAgentState:
        self.logger.info(f"Running Mode C (mixed) with {len(tasks)} task(s)")
        return LegalAgentState(
            tasks=tasks,
            context=context,
            deterministic_outputs=deterministic_outputs or {},
        )

    def build_graph(self) -> Any:
        return build_legal_graph(self)

    def run_react_agent(
        self,
        tasks: list[str],
        deterministic_outputs: dict[str, ToolResult[Any]],
        context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, list[str]]:
        """Run a ReAct agent with the optional tools bound.

        Prepends deterministic results as context so the LLM doesn't try to
        re-fetch them.
        """
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

        user_prompt = build_user_prompt(tasks, render_tool_outputs(deterministic_outputs))
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": self.config.max_llm_iterations},
        )
        return result, self.extract_final_answer(result), self.extract_used_tools(result)
