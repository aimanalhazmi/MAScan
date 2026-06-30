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
from mascan.agents.sources import (
    dedupe_sources,
    render_source_lines,
    sources_from_tool_results,
)
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model

class PoliticalAgent(BaseAgent):
    name = "political"  # must match config.yaml `name`

    def _run(self, tasks: list[str], context: dict[str, Any] | None = None, deterministic_outputs: dict[str, ToolResult[Any]] | None = None) -> AgentReport:
        self.logger.info(f"Running Mode C (mixed) with {len(tasks)} task(s)")

        # LLM with optional tools — decides what else (if anything) to call.
        findings, llm_used_tools = self.run_react_agent(
            tasks,
            deterministic_outputs,
            context=context,
        )

        # assemble the report.
        sources = self.collect_sources(deterministic_outputs, llm_used_tools)
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

    def get_optional_tools(self) -> list[Any]:
        """Return LangChain-wrapped tools the LLM is allowed to call."""
        return [
            tool.as_langchain_tool()
            for name, tool in self.optional_tools.items()
            if name in self.config.optional_tools
        ]

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
        return self.extract_final_answer(result), self.extract_used_tools(result)

    @staticmethod
    def extract_final_answer(result: dict[str, Any]) -> str:
        """Last message in the ReAct result is the LLM's final answer."""
        messages = result.get("messages", [])
        if not messages:
            return "(no response)"
        return str(messages[-1].content)

    @staticmethod
    def extract_used_tools(result: dict[str, Any]) -> list[str]:
        """Walk the message history and collect every tool the LLM invoked."""
        used: list[str] = []
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name and name not in used:
                    used.append(name)
        return used

    def collect_sources(
        self,
        deterministic_outputs: dict[str, ToolResult[Any]],
        llm_used_tools: list[str],
    ) -> list[Source]:
        """Real article links from the always-called tools (web + news)."""
        return dedupe_sources(sources_from_tool_results(deterministic_outputs))

    def render_markdown(
        self,
        tasks: list[str],
        findings: str,
        sources: list[Source],
        llm_used_tools: list[str],
    ) -> str:
        task_lines = "\n".join(f"- {t}" for t in tasks)
        src_lines = render_source_lines(sources)
        llm_lines = "\n".join(f"- {t}" for t in llm_used_tools) or "- (none)"
        always_lines = "\n".join(f"- {t}" for t in self.config.always_call_tools)
        return (
            "## Political Analysis\n\n"
            f"**Tasks:**\n{task_lines}\n\n"
            f"**Findings:**\n\n{findings}\n\n"
            f"**Tools always called:**\n{always_lines}\n\n"
            f"**Tools the LLM chose to call:**\n{llm_lines}\n\n"
            f"**Sources:**\n{src_lines}\n"
        )
