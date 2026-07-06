"""PoliticalAgent — Mode C (mixed).

Pattern:
  1. Always call certain tools deterministically (core data we always need).
  2. Pass the results as context to a ReAct agent that decides whether to
     also call optional tools, then writes the final answer.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from mascan.agents.base import BaseAgent
from mascan.agents.context import render_tool_outputs
from mascan.agents.political.prompts import build_user_prompt
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model

class PoliticalAgent(BaseAgent):
    name = "political"  # must match config.yaml `name`

    def _run(self, tasks: list[str], context: dict[str, Any] | None = None, deterministic_outputs: dict[str, ToolResult[Any]] | None = None) -> AgentReport:
        self.logger.info(f"Running Mode C (mixed) with {len(tasks)} task(s)")

        # LLM with optional tools — decides what else (if anything) to call.
        result, findings, llm_used_tools = self.run_react_agent(
            tasks,
            deterministic_outputs,
            context=context,
        )

        # assemble the report.
        sources = self.collect_sources(deterministic_outputs=deterministic_outputs, react_result=result)
        rendered = self.render_markdown(tasks, findings, sources, llm_used_tools)

        return AgentReport(
            agent_name=self.name,
            tasks=tasks,
            findings=findings,
            sources=sources,
            confidence=0.7,
            rendered_markdown=rendered,
            metadata={
                "mode": "mixed",
                "deterministic_tools": list(self.config.always_call_tools),
                "llm_chosen_tools": llm_used_tools,
            },
        )

    def run_react_agent(
        self,
        tasks: list[str],
        deterministic_outputs: dict[str, ToolResult[Any]],
        context: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
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

        # Runtime context is rendered into the prompt so the LLM can resolve relative dates.
        user_prompt = build_user_prompt(
            tasks,
            render_tool_outputs(deterministic_outputs),
            context=context,
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": self.config.max_llm_iterations},
        )
        return result,self.extract_final_answer(result), self.extract_used_tools(result)
