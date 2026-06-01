import argparse
import sys

import mascan.agents.economics
import mascan.agents.political         
import mascan.agents.social_media

from mascan.core.logging import configure_logging, get_logger
from mascan.orchestrator.graph import run, stream


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MAScan orchestrator.")
    parser.add_argument("query", help="The user query.")
    parser.add_argument("--stream", action="store_true", help="Stream node updates.")
    args = parser.parse_args()

    configure_logging()
    logger = get_logger("scripts.run_orchestrator")
    logger.info("Query: %r (stream=%s)", args.query, args.stream)

    if args.stream:
        for event in stream(args.query):
            node = event["node"]
            update = event["update"]
            print(f"\n--- node: {node} ---")
            for key, value in update.items():
                preview = str(value)
                if len(preview) > 300:
                    preview = preview[:300] + "..."
                print(f"  {key}: {preview}")
        return 0

    report = run(args.query)
    print("\n" + "=" * 70)
    print(report.rendered_markdown)
    print("=" * 70)
    print(f"\nAgents run: {sorted(report.agent_reports.keys())}")
    if report.failures:
        print(f"Failures:   {report.failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
