"""EconomicsAgent — Mode C (mixed).

Pattern:
  1. Always call certain tools deterministically (core data we always need).
  2. Pass the results as context to a ReAct agent that decides whether to
     also call optional tools, then writes the final answer.
"""

import ast
import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from mascan.agents.base import BaseAgent
from mascan.agents.context import render_tool_outputs
from mascan.agents.economics.prompts import build_user_prompt
from mascan.agents.sources import (
    dedupe_sources,
    render_source_lines,
    sources_from_react,
    sources_from_tool_results,
)
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model

ALWAYS_CALL_TOOLS: tuple[str, ...] = ("web_search",)  # called every run
OPTIONAL_TOOLS: tuple[str, ...] = ("get_weekly_stock_prices", "rag_search")  # LLM may call
MAX_LLM_ITERATIONS = 10  # passed to create_react_agent as recursion_limit


class EconomicsAgent(BaseAgent):
    name = "economics"  # must match config.yaml `name`

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        self.logger.info("Running Mode C (mixed) with %d task(s)", len(tasks))

        # deterministic tools — always called.
        deterministic_outputs = self.gather_deterministic(tasks)

        # LLM with optional tools — decides what else (if anything) to call.
        findings, llm_used_tools, llm_sources = self.run_react_agent(
            tasks,
            deterministic_outputs,
            context=context,
        )

        # assemble the report.
        sources = self.collect_sources(deterministic_outputs, llm_sources)
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
        """Call the always-call tools regardless of the question."""
        query = " ; ".join(tasks)
        outputs: dict[str, ToolResult] = {}
        for tool_name in ALWAYS_CALL_TOOLS:
            if tool_name in self.tools:
                outputs[tool_name] = self.tools[tool_name].run(query=query)
            else:
                self.logger.warning("Always-call tool %r not available; skipping.", tool_name)
        return outputs

    def get_optional_tools(self) -> list:
        """Return LangChain-wrapped tools the LLM is allowed to call."""
        return [
            tool.as_langchain_tool()
            for name, tool in self.tools.items()
            if name in OPTIONAL_TOOLS
        ]

    def run_react_agent(
        self,
        tasks: list[str],
        deterministic_outputs: dict[str, ToolResult],
        context: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], list[Source]]:
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
            config={"recursion_limit": MAX_LLM_ITERATIONS},
        )
        # yfinance calls expose no URL in their payload, so reconstruct a Yahoo
        # Finance link from the ticker; other tools yield real article links.
        llm_sources = dedupe_sources(
            self.extract_llm_sources(result) + sources_from_react(result)
        )
        return (
            self.extract_final_answer(result),
            self.extract_used_tools(result),
            llm_sources,
        )

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

    @classmethod
    def extract_llm_sources(cls, result: dict[str, Any]) -> list[Source]:
        """Turn the LLM's market-data tool calls into rich Sources.

        ``get_weekly_stock_prices`` returns only its ``data`` payload to the LLM
        (the ``yfinance:TICKER`` source on the ToolResult is dropped by
        ``as_langchain_tool``), so we reconstruct the source from the payload:
        the ticker, the resolved company name, and a Yahoo Finance link.
        """
        sources_by_name: dict[str, Source] = {}
        for msg in result.get("messages", []):
            if getattr(msg, "type", None) != "tool":
                continue
            if getattr(msg, "name", None) != "get_weekly_stock_prices":
                continue
            payload = cls._parse_tool_payload(msg.content)
            if not isinstance(payload, dict):
                continue
            ticker = payload.get("ticker")
            if not isinstance(ticker, str) or not ticker:
                continue
            fundamentals = payload.get("fundamentals")
            company = (
                fundamentals.get("company_name")
                if isinstance(fundamentals, dict)
                else None
            )
            name = f"yfinance:{ticker}"
            if name in sources_by_name:
                continue
            sources_by_name[name] = Source(
                name=name,
                url=f"https://finance.yahoo.com/quote/{ticker}",
                metadata={
                    "used_by": "llm_decision",
                    "tool": "get_weekly_stock_prices",
                    "provider": "yfinance",
                    "ticker": ticker,
                    "company_name": company,
                    "start_date": payload.get("start_date"),
                    "end_date": payload.get("end_date"),
                },
            )
        return list(sources_by_name.values())

    @staticmethod
    def _parse_tool_payload(content: Any) -> Any:
        """Best-effort decode of a ToolMessage payload (JSON, dict-repr, or dict)."""
        if isinstance(content, dict):
            return content
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    def collect_sources(
        self,
        deterministic_outputs: dict[str, ToolResult],
        llm_sources: list[Source],
    ) -> list[Source]:
        """Merge real article links from both the deterministic and LLM paths."""
        return dedupe_sources(
            sources_from_tool_results(deterministic_outputs) + llm_sources
        )

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
            f"## {self.name.title()} Analysis\n\n"
            f"**Tasks:**\n{task_lines}\n\n"
            f"**Findings:**\n\n{findings}\n\n"
            f"**Tools always called:**\n{always_lines}\n\n"
            f"**Tools the LLM chose to call:**\n{llm_lines}\n\n"
            f"**Sources:**\n{src_lines}\n"
        )
