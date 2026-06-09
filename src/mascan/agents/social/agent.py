"""SocialAgent — Mode C (mixed)."""

import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mascan.agents.base import BaseAgent
from mascan.agents.social.prompts import build_user_prompt, render_tool_outputs
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model

ALWAYS_CALL_TOOLS: tuple[str, ...] = ("web_search", "world_bank_social_indicators")
OPTIONAL_TOOLS: tuple[str, ...] = ("reddit_search", "x_search")
MAX_LLM_ITERATIONS = 10
MAX_SEARCH_QUERIES = 3
WEB_RESULTS_PER_QUERY = 5
REDDIT_RESULTS_PER_QUERY = 10
X_RESULTS_PER_QUERY = 10

SOCIAL_EVIDENCE_PLANNER_PROMPT = """\
You plan evidence collection for the Social analyst in a PESTEL market-analysis system.

Given the assigned social-analysis tasks:
1. Decide which World Bank country codes are relevant. Use ISO-3 / World Bank codes
   such as DEU, USA, CHN, GBR, EUU, or WLD. Pick WLD only when no specific geography
   is relevant.
2. Write up to 3 public web search queries. Do not simply copy the task; make each
   query targeted at social evidence such as consumer sentiment, workforce, education,
   public health, inequality, adoption barriers, social controversy, or community risk.
3. Write up to 3 Reddit search queries only when Reddit discussion could add useful
   qualitative evidence. Keep them short keyword queries.
4. Write up to 3 X search queries only when recent public posts could add useful
   qualitative evidence. Keep them short keyword queries.

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
    reddit_queries: list[str] = Field(
        default_factory=list,
        description="Targeted Reddit search queries for qualitative evidence.",
    )
    x_queries: list[str] = Field(
        default_factory=list,
        description="Targeted X search queries for recent qualitative evidence.",
    )


class SocialAgent(BaseAgent):
    name = "social"

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        self.logger.info("Running Mode C (mixed) with %d task(s)", len(tasks))

        deterministic_outputs = self.gather_deterministic(tasks)
        findings, llm_used_tools = self.run_react_agent(tasks, deterministic_outputs)
        evidence_tools = self.extract_evidence_tools(deterministic_outputs)
        sources = self.collect_sources(deterministic_outputs, llm_used_tools)
        rendered = self.render_markdown(tasks, findings, sources, evidence_tools)

        return AgentReport(
            agent_name=self.name,
            tasks=tasks,
            findings=findings,
            sources=sources,
            confidence=0.65,
            rendered_markdown=rendered,
            metadata={
                "mode": "mixed",
                "deterministic_tools": list(ALWAYS_CALL_TOOLS),
                "llm_chosen_tools": llm_used_tools,
                "evidence_tools": evidence_tools,
                "evidence_plan": getattr(self, "_last_evidence_plan", None),
            },
        )

    def gather_deterministic(self, tasks: list[str]) -> dict[str, ToolResult[Any]]:
        query = " ; ".join(tasks)
        plan = self.plan_evidence(tasks)
        self._last_evidence_plan = plan.model_dump()
        outputs: dict[str, ToolResult[Any]] = {}

        if "web_search" in self.tools:
            outputs.update(
                self.run_query_batch(
                    tool_name="web_search",
                    queries=plan.web_queries or [query],
                    limit_kwarg="max_results",
                    limit=WEB_RESULTS_PER_QUERY,
                )
            )
        else:
            self.logger.warning("Always-call tool %r not available; skipping.", "web_search")

        if "world_bank_social_indicators" in self.tools:
            outputs["world_bank_social_indicators"] = self.tools[
                "world_bank_social_indicators"
            ].run(country_codes=self.normalize_country_codes(plan.country_codes))
        else:
            self.logger.warning(
                "Always-call tool %r not available; skipping.",
                "world_bank_social_indicators",
            )

        if self.env_enabled("SOCIAL_ENABLE_REDDIT") and "reddit_search" in self.tools:
            outputs.update(
                self.run_query_batch(
                    tool_name="reddit_search",
                    queries=plan.reddit_queries,
                    limit_kwarg="limit",
                    limit=REDDIT_RESULTS_PER_QUERY,
                )
            )

        if self.env_enabled("SOCIAL_ENABLE_X") and "x_search" in self.tools:
            outputs.update(
                self.run_query_batch(
                    tool_name="x_search",
                    queries=plan.x_queries,
                    limit_kwarg="max_results",
                    limit=X_RESULTS_PER_QUERY,
                )
            )
        return outputs

    @staticmethod
    def env_enabled(name: str, default: bool = True) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def plan_evidence(self, tasks: list[str]) -> SocialEvidencePlan:
        llm = get_chat_model(
            model=self.config.model,
            temperature=0.0,
            max_tokens=1000,
        )
        structured_llm = llm.with_structured_output(SocialEvidencePlan)
        result: SocialEvidencePlan = structured_llm.invoke(
            [
                SystemMessage(content=SOCIAL_EVIDENCE_PLANNER_PROMPT),
                HumanMessage(content="\n".join(f"- {task}" for task in tasks)),
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
            reddit_queries=cls.clean_queries(plan.reddit_queries),
            x_queries=cls.clean_queries(plan.x_queries),
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
        return [
            tool.as_langchain_tool()
            for name, tool in self.tools.items()
            if name in OPTIONAL_TOOLS
        ]

    def run_react_agent(
        self,
        tasks: list[str],
        deterministic_outputs: dict[str, ToolResult[Any]],
    ) -> tuple[str, list[str]]:
        llm = get_chat_model(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=self.config.system_prompt,
        )

        user_prompt = build_user_prompt(tasks, render_tool_outputs(deterministic_outputs))
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": MAX_LLM_ITERATIONS},
        )
        return self.extract_final_answer(result), self.extract_used_tools(result)

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

    def collect_sources(
        self,
        deterministic_outputs: dict[str, ToolResult[Any]],
        llm_used_tools: list[str],
    ) -> list[Source]:
        sources_by_name: dict[str, Source] = {}
        for result in deterministic_outputs.values():
            if result.success:
                source_urls = result.metadata.get("source_urls")
                url = source_urls[0] if isinstance(source_urls, list) and source_urls else None
                if result.source in sources_by_name:
                    sources_by_name[result.source] = self.merge_source_metadata(
                        sources_by_name[result.source],
                        result.metadata,
                    )
                else:
                    sources_by_name[result.source] = Source(
                        name=result.source,
                        url=url,
                        metadata=result.metadata,
                    )
        for name in llm_used_tools:
            if name not in sources_by_name:
                sources_by_name[name] = Source(name=name, metadata={"used_by": "llm_decision"})
        return list(sources_by_name.values())

    @staticmethod
    def merge_source_metadata(source: Source, metadata: dict[str, Any]) -> Source:
        merged = dict(source.metadata)
        if isinstance(merged.get("query"), str):
            merged["queries"] = [merged["query"]]

        for key in ("source_urls", "queries"):
            existing = merged.get(key)
            values = existing if isinstance(existing, list) else []
            new_values = metadata.get(key)
            if isinstance(new_values, list):
                values = [*values, *new_values]
            elif key == "queries" and isinstance(metadata.get("query"), str):
                values = [*values, metadata["query"]]
            if values:
                merged[key] = list(dict.fromkeys(values))

        if isinstance(metadata.get("query"), str):
            queries = merged.get("queries") if isinstance(merged.get("queries"), list) else []
            merged["queries"] = list(dict.fromkeys([*queries, metadata["query"]]))

        count = merged.get("count", 0)
        if isinstance(count, int) and isinstance(metadata.get("count"), int):
            merged["count"] = count + metadata["count"]

        return Source(name=source.name, url=source.url, metadata=merged)

    @staticmethod
    def extract_evidence_tools(outputs: dict[str, ToolResult[Any]]) -> list[str]:
        tools: list[str] = []
        for name, result in outputs.items():
            if not result.success:
                continue
            tool_name = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name
            if tool_name not in tools:
                tools.append(tool_name)
        return tools

    def render_markdown(
        self,
        tasks: list[str],
        findings: str,
        sources: list[Source],
        evidence_tools: list[str],
    ) -> str:
        task_lines = "\n".join(f"- {t}" for t in tasks)
        src_lines = "\n".join(self.format_source_line(source) for source in sources) or "- (none)"
        evidence_lines = "\n".join(f"- {t}" for t in evidence_tools) or "- (none)"
        always_lines = "\n".join(f"- {t}" for t in ALWAYS_CALL_TOOLS)
        return (
            "## Social Analysis\n\n"
            f"**Tasks:**\n{task_lines}\n\n"
            f"**Findings:**\n\n{findings}\n\n"
            f"**Tools always called:**\n{always_lines}\n\n"
            f"**Evidence tools selected:**\n{evidence_lines}\n\n"
            f"**Sources:**\n{src_lines}\n"
        )

    @staticmethod
    def format_source_line(source: Source) -> str:
        source_urls = source.metadata.get("source_urls")
        if isinstance(source_urls, list) and source_urls:
            links = ", ".join(f"[{index + 1}]({url})" for index, url in enumerate(source_urls))
            return f"- {source.name}: {links}"

        if source.url:
            return f"- [{source.name}]({source.url})"

        return f"- {source.name}"
