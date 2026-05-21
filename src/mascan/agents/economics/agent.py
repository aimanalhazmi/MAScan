"""EconomicsAgent — Mode C (mixed) template.

Pattern:
  1. Always call certain tools deterministically (core data you always need).
  2. Let the LLM decide whether to also call optional tools.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from mascan.agents.base import BaseAgent
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model
from mascan.agents.economics.prompts import render_tool_outputs, build_user_prompt

ALWAYS_CALL_TOOLS: tuple[str, ...] = ("tool_name_1",)  # called every run
OPTIONAL_TOOLS: tuple[str, ...] = ("tool_name_2",)     # LLM may call
MAX_LLM_TOOL_ROUNDS = 3                                # loop safety cap


class EconomicsAgent(BaseAgent):
    name = "economics"  # must match config.yaml `name`

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        self.logger.info("Running Mode C (mixed) with %d task(s)", len(tasks))

        deterministic_outputs = self.gather_deterministic(tasks)


        findings, llm_used_tools = self.synthesize_with_optional_tools(
            tasks=tasks,
            deterministic_outputs=deterministic_outputs,
        )

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
                "deterministic_tools": list(ALWAYS_CALL_TOOLS),
                "llm_chosen_tools": llm_used_tools,
            },
        )

    def gather_deterministic(self, tasks: list[str]) -> dict[str, ToolResult]:
        query = " ; ".join(tasks)
        outputs: dict[str, ToolResult] = {}
        for tool_name in ALWAYS_CALL_TOOLS:
            if tool_name in self.tools:
                outputs[tool_name] = self.tools[tool_name].run(query=query)
            else:
                self.logger.warning("Always-call tool %r not available; skipping.", tool_name)
        return outputs

    def synthesize_with_optional_tools(
        self,
        tasks: list[str],
        deterministic_outputs: dict[str, ToolResult],
    ) -> tuple[str, list[str]]:
        optional_tools = {
            name: tool for name, tool in self.tools.items() if name in OPTIONAL_TOOLS
        }

        llm = get_chat_model(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if optional_tools:
            llm = llm.bind_tools([t.as_langchain_tool() for t in optional_tools.values()])

        tool_block = render_tool_outputs(deterministic_outputs)
        user_prompt = build_user_prompt(tasks, tool_block)

        messages: list[Any] = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        llm_used_tools: list[str] = []

        for round_idx in range(MAX_LLM_TOOL_ROUNDS):
            response = llm.invoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return str(response.content), llm_used_tools

            self.logger.info(
                "LLM round %d requested %d tool call(s)", round_idx + 1, len(tool_calls)
            )

            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else call.name
                args = call.get("args") if isinstance(call, dict) else call.args
                call_id = call.get("id") if isinstance(call, dict) else call.id

                if name not in optional_tools:
                    tool_content = f"Tool {name!r} is not available to this agent."
                else:
                    result = optional_tools[name].run(**(args or {}))
                    if name not in llm_used_tools:
                        llm_used_tools.append(name)
                    tool_content = (
                        str(result.data)
                        if result.success
                        else f"Tool {name} failed: {result.error}"
                    )

                messages.append(ToolMessage(content=tool_content, tool_call_id=call_id))

        self.logger.warning("Hit MAX_LLM_TOOL_ROUNDS=%d, forcing final answer", MAX_LLM_TOOL_ROUNDS)
        messages.append(HumanMessage(content="Please provide your final answer now."))
        final = llm.invoke(messages)
        return str(final.content), llm_used_tools


    def collect_sources(
        self,
        deterministic_outputs: dict[str, ToolResult],
        llm_used_tools: list[str],
    ) -> list[Source]:
        sources: list[Source] = []
        for result in deterministic_outputs.values():
            if result.success:
                sources.append(Source(name=result.source, metadata=result.metadata))
        for name in llm_used_tools:
            sources.append(Source(name=name, metadata={"used_by": "llm_decision"}))
        return sources

    def render_markdown(
        self,
        tasks: list[str],
        findings: str,
        sources: list[Source],
        llm_used_tools: list[str],
    ) -> str:
        task_lines = "\n".join(f"- {t}" for t in tasks)
        src_lines = "\n".join(f"- {s.name}" for s in sources) or "- (none)"
        llm_lines = "\n".join(f"- {t}" for t in llm_used_tools) or "- (none)"
        always_lines = "\n".join(f"- {t}" for t in ALWAYS_CALL_TOOLS)
        return (
            f"## {self.name.title()} Analysis\n\n"
            f"**Tasks:**\n{task_lines}\n\n"
            f"**Findings:**\n\n{findings}\n\n"
            f"**Tools always called:**\n{always_lines}\n\n"
            f"**Tools the LLM chose to call:**\n{llm_lines}\n\n"
            f"**Sources:**\n{src_lines}\n"
        )