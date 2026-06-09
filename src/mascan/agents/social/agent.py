"""SocialAgent — Mode C (mixed)."""

import json
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mascan.agents.base import BaseAgent
from mascan.agents.context import render_agent_context, render_runtime_context, render_tool_outputs
from mascan.agents.social.prompts import build_user_prompt
from mascan.contracts.reports import AgentReport, Source
from mascan.contracts.tools import ToolResult
from mascan.core.llm import get_chat_model

ALWAYS_CALL_TOOLS: tuple[str, ...] = ("web_search", "world_bank_social_indicators")
OPTIONAL_TOOLS: tuple[str, ...] = ("reddit_search", "x_search")
MAX_LLM_ITERATIONS = 10
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

    def run(self, tasks: list[str], context: dict[str, Any] | None = None) -> AgentReport:
        self.logger.info("Running Mode C (mixed) with %d task(s)", len(tasks))

        deterministic_outputs = self.gather_deterministic(tasks, context=context)
        findings, llm_used_tools, llm_sources = self.run_react_agent(
            tasks,
            deterministic_outputs,
            context=context,
        )
        sources = self.collect_sources(deterministic_outputs, llm_sources)
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
                "deterministic_tools": list(ALWAYS_CALL_TOOLS),
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
            if name in OPTIONAL_TOOLS and enabled.get(name, True)
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
            config={"recursion_limit": MAX_LLM_ITERATIONS},
        )
        return (
            self.extract_final_answer(result),
            self.extract_used_tools(result),
            self.extract_llm_sources(result),
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

    @classmethod
    def extract_llm_sources(cls, result: dict[str, Any]) -> list[Source]:
        """Turn the LLM's tool-call outputs (ToolMessages) into Sources with post links."""
        urls_by_tool: dict[str, list[str]] = {}
        for msg in result.get("messages", []):
            if getattr(msg, "type", None) != "tool":
                continue
            name = getattr(msg, "name", None)
            if not name:
                continue
            bucket = urls_by_tool.setdefault(name, [])
            for url in cls._extract_urls(msg.content):
                if url not in bucket:
                    bucket.append(url)

        sources: list[Source] = []
        for name, urls in urls_by_tool.items():
            sources.append(
                Source(
                    name=name,
                    url=urls[0] if urls else None,
                    metadata={
                        "used_by": "llm_decision",
                        "source_urls": urls,
                        "count": len(urls),
                    },
                )
            )
        return sources

    @staticmethod
    def _extract_urls(content: Any) -> list[str]:
        """Collect post URLs from a ToolMessage payload (JSON list of dicts, or text)."""
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        urls: list[str] = []
        try:
            data: Any = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            data = None

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                url = obj.get("url")
                if isinstance(url, str) and url.startswith("http"):
                    urls.append(url)
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        if data is not None:
            walk(data)
        if not urls:
            urls = re.findall(r"https?://[^\s\"'\)\]]+", text)
        return list(dict.fromkeys(urls))

    def collect_sources(
        self,
        deterministic_outputs: dict[str, ToolResult[Any]],
        llm_sources: list[Source],
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
        for source in llm_sources:
            if source.name in sources_by_name:
                sources_by_name[source.name] = self.merge_source_metadata(
                    sources_by_name[source.name],
                    source.metadata,
                )
            else:
                sources_by_name[source.name] = source
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

    def render_markdown(
        self,
        tasks: list[str],
        findings: str,
        sources: list[Source],
        llm_used_tools: list[str],
    ) -> str:
        task_lines = "\n".join(f"- {t}" for t in tasks)
        src_lines = "\n".join(self.format_source_line(source) for source in sources) or "- (none)"
        always_lines = "\n".join(f"- {t}" for t in ALWAYS_CALL_TOOLS)
        llm_lines = "\n".join(f"- {t}" for t in llm_used_tools) or "- (none)"
        return (
            "## Social Analysis\n\n"
            f"**Tasks:**\n{task_lines}\n\n"
            f"**Findings:**\n\n{findings}\n\n"
            f"**Tools always called:**\n{always_lines}\n\n"
            f"**Tools the LLM chose to call:**\n{llm_lines}\n\n"
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
