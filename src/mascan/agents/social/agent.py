"""SocialAgent — Mode C (mixed)."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mascan.agents.base import BaseAgent
from mascan.agents.context import render_agent_context, render_runtime_context, render_tool_outputs
from mascan.agents.social.prompts import build_user_prompt
from mascan.agents.sources import (
    dedupe_sources,
    render_source_lines,
    sources_from_react,
    sources_from_tool_results,
)
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model

MAX_SEARCH_QUERIES = 3
WEB_RESULTS_PER_QUERY = 5

SOCIAL_EVIDENCE_PLANNER_PROMPT = """\
You plan evidence collection for the Social analyst in a PESTEL market-analysis system.

Given the assigned social-analysis tasks:
1. Decide which World Bank country codes are relevant. Use ISO-3 / World Bank codes
   such as DEU, USA, CHN, GBR, EUU, or WLD. Pick WLD only when no specific geography
   is relevant.
2. Write up to 3 public web search queries. Do not simply copy the task; make each
   query targeted at social evidence such as consumer sentiment, workforce, education,
   public health, inequality, adoption barriers, social controversy, or community risk.

Return concise search phrases. Prefer fewer high-quality queries over filling all slots.
"""


class SocialEvidencePlan(BaseModel):
    """Structured plan for SocialAgent evidence gathering."""

    country_codes: list[str] = Field(
        default_factory=lambda: ["WLD"],
        description="World Bank ISO-3 country codes to query.",
    )
    web_queries: list[str] = Field(
        default_factory=list,
        description="Targeted public-web search queries.",
    )


class SocialAgent(BaseAgent):
    name = "social"

    def _run(self, tasks: list[str], context: dict[str, Any] | None = None, deterministic_outputs: dict[str, ToolResult[Any]] | None = None) -> AgentReport:
        self.logger.info(f"Running Mode C (mixed) with {len(tasks)} task(s)")

        result, findings, llm_used_tools, llm_sources = self.run_react_agent(
            tasks,
            deterministic_outputs,
            context=context,
        )
        sources = self.collect_sources(deterministic_outputs=deterministic_outputs, react_result=result)
        rendered = self.render_markdown(tasks, findings, sources, llm_used_tools)

        return AgentReport(
            agent_name=self.name,
            tasks=tasks,
            findings=findings,
            sources=sources,
            confidence=0.65,
            rendered_markdown=rendered,
            metadata={
                "mode": "mixed",
                "deterministic_tools": list(self.config.always_call_tools),
                "llm_chosen_tools": llm_used_tools,
                "evidence_plan": getattr(self, "_last_evidence_plan", None),
            },
        )

    def gather_deterministic(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, ToolResult[Any]]:
        query = " ; ".join(tasks)
        plan = self.plan_evidence(tasks, context=context)
        self._last_evidence_plan = plan.model_dump()
        outputs: dict[str, ToolResult[Any]] = {}

        self.tools = {**self.always_call_tools, **self.optional_tools}  # merge for this method only TODO: fix this hack

        # if "web_search" in self.tools:
        #     outputs.update(
        #         self.run_query_batch(
        #             tool_name="web_search",
        #             queries=plan.web_queries or [query],
        #             limit_kwarg="max_results",
        #             limit=WEB_RESULTS_PER_QUERY,
        #         )
        #     )
        # else:
        #     self.logger.warning("Always-call tool %r not available; skipping.", "web_search")

        if "world_bank_social_indicators" in self.tools:
            outputs["world_bank_social_indicators"] = self.tools[
                "world_bank_social_indicators"
            ].run(country_codes=self.normalize_country_codes(plan.country_codes))
        else:
            self.logger.warning(
                "Always-call tool %r not available; skipping.",
                "world_bank_social_indicators",
            )

        return outputs

    def plan_evidence(
        self,
        tasks: list[str],
        context: dict[str, Any] | None = None,
    ) -> SocialEvidencePlan:
        llm = get_chat_model(
            model=self.config.model,
            temperature=0.0,
            max_tokens=1000,
        )
        structured_llm = llm.with_structured_output(SocialEvidencePlan)
        result: SocialEvidencePlan = structured_llm.invoke(
            [
                SystemMessage(content=SOCIAL_EVIDENCE_PLANNER_PROMPT),
                HumanMessage(
                    content=(
                        f"{render_agent_context(context)}"
                        f"{render_runtime_context(context)}"
                        "Assigned social-analysis tasks:\n"
                        + "\n".join(f"- {task}" for task in tasks)
                    )
                ),
            ]
        )
        return self.constrain_evidence_plan(result)

    def run_query_batch(
        self,
        tool_name: str,
        queries: list[str],
        limit_kwarg: str,
        limit: int,
    ) -> dict[str, ToolResult[Any]]:
        outputs: dict[str, ToolResult[Any]] = {}
        for index, query in enumerate(self.clean_queries(queries), start=1):
            outputs[f"{tool_name}_{index}"] = self.tools[tool_name].run(
                query=query,
                **{limit_kwarg: limit},
            )
        return outputs

    @classmethod
    def constrain_evidence_plan(cls, plan: SocialEvidencePlan) -> SocialEvidencePlan:
        return SocialEvidencePlan(
            country_codes=cls.normalize_country_codes(plan.country_codes),
            web_queries=cls.clean_queries(plan.web_queries),
        )

    @staticmethod
    def normalize_country_codes(country_codes: list[str]) -> list[str]:
        normalized: list[str] = []
        for code in country_codes:
            cleaned = code.strip().upper()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized[:MAX_SEARCH_QUERIES] or ["WLD"]

    @staticmethod
    def clean_queries(queries: list[str]) -> list[str]:
        cleaned: list[str] = []
        for query in queries:
            normalized = " ".join(query.split())
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned[:MAX_SEARCH_QUERIES]

    def get_optional_tools(self) -> list[Any]:
        """LangChain-wrapped optional tools the LLM may call, gated by config.yaml options."""
        options = self.config.options
        enabled = {
            "reddit_search": options.get("enable_reddit", True),
            "x_search": options.get("enable_x", True),
        }
        return [
            tool.as_langchain_tool()
            for name, tool in self.tools.items()
            if name in self.config.optional_tools and enabled.get(name, True)
        ]

    def run_react_agent(
        self,
        tasks: list[str],
        deterministic_outputs: dict[str, ToolResult[Any]],
        context: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], list[Source]]:
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
            render_tool_outputs(deterministic_outputs),
            context=context,
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": self.config.max_llm_iterations},
        )
        return (
            self.extract_final_answer(result),
            self.extract_used_tools(result),
            sources_from_react(result),
        )

    @staticmethod
    def extract_final_answer(result: dict[str, Any]) -> str:
        messages = result.get("messages", [])
        if not messages:
            return "(no response)"
        return str(messages[-1].content)

    @staticmethod
    def extract_used_tools(result: dict[str, Any]) -> list[str]:
        used: list[str] = []
        for msg in result.get("messages", []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name and name not in used:
                    used.append(name)
        return used
