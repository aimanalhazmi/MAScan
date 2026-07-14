from typing import Any

from langchain.agents import create_agent

from mascan.agents import BaseAgent
from mascan.agents.environmental.prompts import build_user_prompt
from mascan.contracts.reports import AgentReport
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model


class EnvironmentalAgent(BaseAgent):
    name = "environmental"  # must match config.yaml `name`

    def _run(self, tasks: list[str], context: dict[str, Any] | None = None, deterministic_outputs: dict[str, ToolResult] | None = None) -> AgentReport:
        self.logger.info(f"Running environmental agent with {len(tasks)} task(s)")

        llm = get_chat_model(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        tools = [tool.as_langchain_tool() for name, tool in self.optional_tools.items()]
        agent = create_agent(model=llm, tools=tools, system_prompt=self.config.system_prompt)

        user_prompt = build_user_prompt(
            tasks,
            context=context,
        )
        result = self.invoke_react_with_fallback(agent, llm, user_prompt)

        findings = self.extract_final_answer(result)
        llm_used_tools = self.extract_used_tools(result)


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
                "mode": "B — LLM-driven",
                "deterministic_tools": [],
                "llm_chosen_tools": llm_used_tools,
            },
        )
