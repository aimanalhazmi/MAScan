"""EconomicsAgent — Mode C (mixed)."""

import ast
import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from mascan.agents.base import BaseAgent
from mascan.agents.context import render_tool_outputs
from mascan.agents.economics.graph import EconomicsAgentState, build_economics_graph
from mascan.agents.economics.prompts import build_user_prompt
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model


class EconomicsAgent(BaseAgent):
    name = "economics"

    def _run(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
        deterministic_outputs: dict[str, ToolResult[Any]] | None = None,
    ) -> AgentReport:
        self.logger.info(f"Running Mode C (mixed) with {len(tasks)} task(s)")

        state = EconomicsAgentState(
            tasks=tasks,
            context=context,
            deterministic_outputs=deterministic_outputs or {},
        )
        final_state = self.build_graph().invoke(state)
        report = final_state.get("report") if isinstance(final_state, dict) else None
        if not isinstance(report, AgentReport):
            raise RuntimeError("Economics graph completed without an AgentReport.")
        return report

    def build_graph(self) -> Any:
        return build_economics_graph(self)

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

        return (
            result,
            self.extract_final_answer(result),
            self.extract_used_tools(result),
        )

    @classmethod
    def extract_llm_sources(cls, result: dict[str, Any]) -> list[Source]:
        """Turn the LLM's market-data tool calls into rich Sources."""
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
