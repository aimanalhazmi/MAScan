from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from mascan.agents import BaseAgent
from mascan.agents.environmental.prompts import build_user_prompt
from mascan.agents.sources import (
    dedupe_sources,
    render_source_lines,
    sources_from_react,
    sources_from_tool_results,
)
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model

ALWAYS_CALL_TOOLS: tuple[str, ...] = ()  # called every run
OPTIONAL_TOOLS: tuple[str, ...] = ()  # LLM may call
MAX_LLM_ITERATIONS = 10  # passed to create_react_agent as recursion_limit

class EnvironmentalAgent(BaseAgent):
    name = "environmental"  # must match config.yaml `name`

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        self.logger.info(f"Running environmental agent with {len(tasks)} task(s)")

        llm = get_chat_model(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        tools = [tool.as_langchain_tool() for name, tool in self.tools.items()]
        agent = create_agent(model=llm, tools=tools, system_prompt=self.config.system_prompt)

        user_prompt = build_user_prompt(
            tasks,
            context=context,
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": MAX_LLM_ITERATIONS},
        )

        findings = self.extract_final_answer(result)
        llm_used_tools = self.extract_used_tools(result)


        # assemble the report.
        sources = self.collect_sources(llm_used_tools, react_result=result)
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


    def get_optional_tools(self) -> list[Any]:
        """Return LangChain-wrapped tools the LLM is allowed to call."""
        return [
            tool.as_langchain_tool()
            for name, tool in self.tools.items()
            if name in OPTIONAL_TOOLS
        ]
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
        llm_used_tools: list[str],
        react_result: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> list[Source]:
        """Real reference links harvested from the agent's tool calls."""
        sources: list[Source] = []
        if deterministic_outputs:
            sources.extend(sources_from_tool_results(deterministic_outputs))
        if react_result:
            sources.extend(sources_from_react(react_result))
        return dedupe_sources(sources)

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
        always_lines = "\n".join(f"- {t}" for t in ALWAYS_CALL_TOOLS)
        return (
            "## Environmental Analysis\n\n"
            f"**Tasks:**\n{task_lines}\n\n"
            f"**Findings:**\n\n{findings}\n\n"
            f"**Tools always called:**\n{always_lines}\n\n"
            f"**Tools the LLM chose to call:**\n{llm_lines}\n\n"
            f"**Sources:**\n{src_lines}\n"
        )
