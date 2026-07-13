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


def _run_mascan(case: GoldStandardCase, *, model: str) -> ModelResponseRecord:
    import mascan.agents.economics  # noqa: F401
    import mascan.agents.environmental  # noqa: F401
    import mascan.agents.legal  # noqa: F401
    import mascan.agents.political  # noqa: F401
    import mascan.agents.social  # noqa: F401
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
    return {
        "runner": "direct_llm",
        "model": model,
        "temperature": 0,
        "max_tokens": 4000,
        "prompt_contract": "gold_standard_pestel_v1",
    }


def main() -> int:
    load_dotenv()
    configure_logging()

    parser = argparse.ArgumentParser(description="Collect gold-standard responses.")
    parser.add_argument(
        "--system",
        choices=["mascan", "zero_shot_same_model", "frontier_model"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--gold-standard", default="eval_papers/gold_standard_cases.json")
    parser.add_argument("--case", action="append", default=None, help="Optional case_id.")
    parser.add_argument("--out", required=True, help="Output JSON list path.")
    args = parser.parse_args()

    dataset = load_gold_standard(args.gold_standard)
    cases = dataset.cases
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case.case_id in selected]
    if not cases:
        raise ValueError("No matching gold-standard cases selected")

    records: list[ModelResponseRecord] = []
    for case in cases:
        try:
            if args.system == "mascan":
                records.append(_run_mascan(case, model=args.model))
            else:
                records.append(_run_llm(case, system_id=args.system, model=args.model))
        except Exception as exc:
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([record.model_dump(mode="json") for record in records], indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
