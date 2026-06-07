"""Run a single agent standalone.

Usage:
    uv run python scripts/run_agent.py economics "EU manufacturing outlook"
    uv run python scripts/run_agent.py political "US-China trade tensions"

Bypasses the orchestrator. Use it to develop and test a single agent
"""

import sys

from dotenv import load_dotenv

from mascan.agents import agent_registry
import mascan.agents.economics  # noqa: F401  # register economics agent
import mascan.agents.political  # noqa: F401  # register political agent
import mascan.agents.environmental  # noqa: F401  # register environmental agent
from mascan.core.logging import configure_logging, get_logger
from mascan.orchestrator.state import RuntimeContext


def main() -> int:
    load_dotenv()
    configure_logging()
    logger = get_logger("scripts.run_agent")

    if len(sys.argv) < 3:
        print(__doc__)
        print(f"\nAvailable agents: {agent_registry.all_names()}")
        return 1

    agent_name = sys.argv[1]
    query = " ".join(sys.argv[2:])

    try:
        agent = agent_registry.get(agent_name)
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Running agent=%s query=%r", agent_name, query)
    runtime_context = RuntimeContext.from_system()
    report = agent.run(
        tasks=[query],
        context={
            "runtime": runtime_context.model_dump(),
        },
    )

    print("\n" + "=" * 70)
    print(report.rendered_markdown)
    print("=" * 70)
    print(f"\nConfidence: {report.confidence}")
    print(f"Sources: {[s.name for s in report.sources]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
