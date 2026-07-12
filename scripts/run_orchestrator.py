import argparse
import sys
from uuid import uuid4

import mascan.agents.economics  # noqa: F401
import mascan.agents.environmental  # noqa: F401
import mascan.agents.legal  # noqa: F401
import mascan.agents.political  # noqa: F401
import mascan.agents.social  # noqa: F401
import mascan.agents.technological  # noqa: F401
from mascan.core.logging import configure_logging, get_logger
from mascan.orchestrator.graph import resume, run, stream


def prompt_clarification(question: str) -> str:
    """Ask the planner's clarification question in the terminal."""
    print("\n[MAScan needs a clarification]")
    print(question)
    try:
        return input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSkipping clarification.")
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MAScan orchestrator.")
    parser.add_argument("query", help="The user query.")
    parser.add_argument("--stream", action="store_true", help="Stream node updates.")
    args = parser.parse_args()

    configure_logging()
    logger = get_logger("scripts.run_orchestrator")
    logger.info("Query: %r (stream=%s)", args.query, args.stream)

    if args.stream:
        thread_id = str(uuid4())
        events = stream(args.query, thread_id)
        while True:
            for event in events:
                if event["node"] == "__interrupt__":
                    answer = prompt_clarification(event["question"])
                    events = resume(thread_id, answer)
                    break
                print(f"\n--- node: {event['node']} ---")
                for key, value in event["update"].items():
                    preview = str(value)
                    if len(preview) > 300:
                        preview = preview[:300] + "..."
                    print(f"  {key}: {preview}")
            else:
                break
        return 0

    report = run(args.query, on_clarify=prompt_clarification)
    print("\n" + "=" * 70)
    print(report.rendered_markdown)
    print("=" * 70)
    print(f"\nAgents run: {sorted(report.agent_reports.keys())}")
    if report.failures:
        print(f"Failures:   {report.failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
