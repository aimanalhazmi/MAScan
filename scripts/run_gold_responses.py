"""Collect model/system responses for the 25-case gold-standard prompt pack."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mascan.core.logging import configure_logging
from mascan.eval.gold_experiment import (
    ModelResponseRecord,
    TokenUsage,
    estimate_token_usage,
    prompt_sha256,
)
from mascan.eval.gold_standard import GoldStandardCase, load_gold_standard


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def _usage_from_message(
    prompt: str,
    response_text: str,
    message: Any,
) -> TokenUsage:
    usage = getattr(message, "usage_metadata", None) or {}
    prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return estimate_token_usage(prompt, response_text)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=(
            total_tokens
            if total_tokens is not None
            else (prompt_tokens or 0) + (completion_tokens or 0)
        ),
        estimated=False,
    )


def _run_llm(case: GoldStandardCase, *, system_id: str, model: str) -> ModelResponseRecord:
    from mascan.core.llm import get_chat_model

    generation_config = {
        "runner": "direct_llm",
        "model": model,
        "temperature": 0,
        "max_tokens": 4000,
        "prompt_contract": "gold_standard_pestel_v1",
    }
    start = time.perf_counter()
    llm = get_chat_model(
        model=model,
        temperature=generation_config["temperature"],
        max_tokens=generation_config["max_tokens"],
    )
    message = llm.invoke(case.prompt)
    latency = time.perf_counter() - start
    response_text = _message_text(message)
    return ModelResponseRecord(
        case_id=case.case_id,
        system_id=system_id,
        model=model,
        prompt_sha256=prompt_sha256(case.prompt),
        generation_config=generation_config,
        response_text=response_text,
        token_usage=_usage_from_message(case.prompt, response_text, message),
        latency_seconds=round(latency, 6),
    )


WEB_SEARCH_SYSTEM_PROMPT = """\
You have a web_search tool and you MUST use it before answering. The case concerns
a specific company, market, and time period, so ground the analysis in retrieved
sources rather than recall alone.

Search first — at least once and at most {budget} times — then stop searching and
write the complete final answer. Follow the output format the user asked for exactly.
"""


def _build_capped_search_tool(budget: int) -> tuple[Any, dict[str, int]]:
    """Wrap web_search with a hard call cap.

    The cap is enforced in code rather than by prompt instruction so the retrieval
    budget of this control is reproducible instead of advisory.
    """
    from langchain_core.tools import StructuredTool

    # Importing the package registers the shared tools using the same settings
    # (including a self-hosted FIRECRAWL_API_URL) that MAScan's own agents use, so
    # this control retrieves through the identical backend.
    import mascan.tools.common  # noqa: F401
    from mascan.tools.common.web_search import WebSearchInput
    from mascan.tools.registry import tool_registry

    search = tool_registry.get("web_search")
    counter = {"calls": 0}

    def run_search(query: str, max_results: int = 5) -> str:
        if counter["calls"] >= budget:
            return (
                "Search budget exhausted. Do not search again. Write the final "
                "answer now using the evidence already gathered."
            )
        counter["calls"] += 1
        result = search.run(query=query, max_results=max_results)
        if not result.success:
            return f"Search failed: {result.error}"
        return json.dumps(result.data)[:20000]

    tool = StructuredTool.from_function(
        func=run_search,
        name="web_search",
        description=search.description,
        args_schema=WebSearchInput,
    )
    return tool, counter


def _usage_from_messages(prompt: str, response_text: str, messages: list[Any]) -> TokenUsage:
    """Sum usage across every model turn in a ReAct run."""
    prompt_tokens = completion_tokens = total_tokens = 0
    seen = False
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            continue
        seen = True
        prompt_tokens += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        completion_tokens += usage.get("output_tokens") or usage.get("completion_tokens") or 0
        total_tokens += usage.get("total_tokens") or 0
    if not seen:
        return estimate_token_usage(prompt, response_text)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        estimated=False,
    )


def _run_llm_with_search(
    case: GoldStandardCase,
    *,
    system_id: str,
    model: str,
    max_searches: int,
) -> ModelResponseRecord:
    """Same model and same frozen prompt as the zero-shot control, plus web search.

    The only manipulated variable relative to `zero_shot_same_model` is internet
    access, so the pair isolates retrieval from everything else.
    """
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphRecursionError

    from mascan.core.llm import get_chat_model

    generation_config = {
        "runner": "direct_llm_web_search",
        "model": model,
        "temperature": 0,
        "max_tokens": 4000,
        "max_searches": max_searches,
        "search_tool": "web_search",
        "prompt_contract": "gold_standard_pestel_v1",
    }
    llm = get_chat_model(model=model, temperature=0, max_tokens=4000)
    tool, counter = _build_capped_search_tool(max_searches)
    agent = create_agent(
        llm,
        [tool],
        system_prompt=WEB_SEARCH_SYSTEM_PROMPT.format(budget=max_searches),
    )

    start = time.perf_counter()
    messages: list[Any] = []
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=case.prompt)]},
            config={"recursion_limit": 2 * max_searches + 6},
        )
        messages = list(result.get("messages") or [])
        response_text = _message_text(messages[-1]) if messages else ""
    except GraphRecursionError:
        # Budget or step cap hit mid-loop: force one tool-free final answer so the
        # case still produces a comparable response instead of an error record.
        final = llm.invoke(
            [
                HumanMessage(content=case.prompt),
                HumanMessage(
                    content=(
                        "Stop searching and write the complete final answer now "
                        "using what you already know."
                    )
                ),
            ]
        )
        messages = [final]
        response_text = _message_text(final)
    latency = time.perf_counter() - start

    generation_config["searches_used"] = counter["calls"]
    return ModelResponseRecord(
        case_id=case.case_id,
        system_id=system_id,
        model=model,
        prompt_sha256=prompt_sha256(case.prompt),
        generation_config=generation_config,
        response_text=response_text,
        token_usage=_usage_from_messages(case.prompt, response_text, messages),
        latency_seconds=round(latency, 6),
    )


def _run_mascan(case: GoldStandardCase, *, model: str) -> ModelResponseRecord:
    import mascan.agents.economics  # noqa: F401
    import mascan.agents.environmental  # noqa: F401
    import mascan.agents.legal  # noqa: F401
    import mascan.agents.political  # noqa: F401
    import mascan.agents.social  # noqa: F401
    import mascan.agents.technological  # noqa: F401
    import mascan.orchestrator.graph as graph_module
    from mascan.agents.registry import agent_registry
    from mascan.core.llm import get_chat_model
    from mascan.core.settings import get_settings
    from mascan.orchestrator.graph import run

    settings = get_settings()
    settings.openai_model_default = model
    for agent in agent_registry.all():
        agent.config = agent.config.model_copy(update={"model": model})
    get_chat_model.cache_clear()
    graph_module.compiled_graph = None

    generation_config = {
        "runner": "mascan_orchestrator",
        "requested_model": model,
        "effective_default_model": settings.openai_model_default,
        "agent_models": {
            agent.name: agent.config.model for agent in agent_registry.all()
        },
        "prompt_contract": "gold_standard_pestel_v1",
    }
    start = time.perf_counter()
    report = run(case.prompt)
    latency = time.perf_counter() - start
    response_text = report.rendered_markdown or report.summary
    return ModelResponseRecord(
        case_id=case.case_id,
        system_id="mascan",
        model=model,
        prompt_sha256=prompt_sha256(case.prompt),
        generation_config=generation_config,
        response_text=response_text,
        token_usage=estimate_token_usage(case.prompt, response_text),
        latency_seconds=round(latency, 6),
    )


def _error_generation_config(system_id: str, model: str) -> dict[str, Any]:
    if system_id == "mascan":
        return {
            "runner": "mascan_orchestrator",
            "requested_model": model,
            "effective_default_model": model,
            "agent_models": {},
            "prompt_contract": "gold_standard_pestel_v1",
        }
    if system_id == "zero_shot_web":
        return {
            "runner": "direct_llm_web_search",
            "model": model,
            "temperature": 0,
            "max_tokens": 4000,
            "search_tool": "web_search",
            "prompt_contract": "gold_standard_pestel_v1",
        }
    return {
        "runner": "direct_llm",
        "model": model,
        "temperature": 0,
        "max_tokens": 4000,
        "prompt_contract": "gold_standard_pestel_v1",
    }


def _load_existing(path: Path) -> dict[str, ModelResponseRecord]:
    """Read already-collected records so a long run can resume after a stall."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    existing: dict[str, ModelResponseRecord] = {}
    for item in payload if isinstance(payload, list) else []:
        try:
            record = ModelResponseRecord.model_validate(item)
        except Exception:  # noqa: BLE001 - a partial file must not block the rerun
            continue
        existing[record.case_id] = record
    return existing


def _write_records(path: Path, records: list[ModelResponseRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([record.model_dump(mode="json") for record in records], indent=2),
        encoding="utf-8",
    )


def main() -> int:
    load_dotenv()
    configure_logging()

    parser = argparse.ArgumentParser(description="Collect gold-standard responses.")
    parser.add_argument(
        "--system",
        choices=[
            "mascan",
            "zero_shot_same_model",
            "zero_shot_web",
            "frontier_model",
        ],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--gold-standard", default="eval_papers/gold_standard_cases.json")
    parser.add_argument("--case", action="append", default=None, help="Optional case_id.")
    parser.add_argument("--out", required=True, help="Output JSON list path.")
    parser.add_argument(
        "--max-searches",
        type=int,
        default=4,
        help=(
            "Hard cap on web_search calls for --system zero_shot_web. Default 4 "
            "models a realistic browsing session; pass 12 to match MAScan's total "
            "retrieval budget (~2 searches x 6 agents)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse successful records already present in --out and only collect "
            "the missing cases. Records are written after every case, so a stalled "
            "or killed run can be resumed without losing completed work."
        ),
    )
    args = parser.parse_args()
    if args.max_searches < 1:
        raise ValueError("--max-searches must be at least 1")

    dataset = load_gold_standard(args.gold_standard)
    cases = dataset.cases
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case.case_id in selected]
    if not cases:
        raise ValueError("No matching gold-standard cases selected")

    out_path = Path(args.out)
    existing = _load_existing(out_path) if args.resume else {}
    records: list[ModelResponseRecord] = []
    for index, case in enumerate(cases, start=1):
        prior = existing.get(case.case_id)
        if prior is not None and not prior.error:
            print(f"[{index}/{len(cases)}] {case.case_id}: reusing existing record", flush=True)
            records.append(prior)
            _write_records(out_path, records)
            continue
        started = time.perf_counter()
        print(f"[{index}/{len(cases)}] {case.case_id}: collecting...", flush=True)
        try:
            if args.system == "mascan":
                records.append(_run_mascan(case, model=args.model))
            elif args.system == "zero_shot_web":
                records.append(
                    _run_llm_with_search(
                        case,
                        system_id=args.system,
                        model=args.model,
                        max_searches=args.max_searches,
                    )
                )
            else:
                records.append(_run_llm(case, system_id=args.system, model=args.model))
        except Exception as exc:
            print(f"    -> FAILED: {type(exc).__name__}: {exc}", flush=True)
            records.append(
                ModelResponseRecord(
                    case_id=case.case_id,
                    system_id=args.system,
                    model=args.model,
                    prompt_sha256=prompt_sha256(case.prompt),
                    generation_config=_error_generation_config(args.system, args.model),
                    error=str(exc),
                )
            )
        else:
            print(f"    -> done in {time.perf_counter() - started:.1f}s", flush=True)

        # Persist after every case so a stall or kill never discards prior work.
        _write_records(out_path, records)

    failed = sum(1 for record in records if record.error)
    print(f"\nCollected {len(records) - failed}/{len(records)} cases ({failed} failed).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
