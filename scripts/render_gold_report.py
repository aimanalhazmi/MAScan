"""Render a Markdown report from gold-standard experiment outputs."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.gold_analysis import SystemComparison
from mascan.eval.gold_experiment import SystemMetricSummary
from mascan.eval.gold_report import render_gold_experiment_report
from mascan.eval.human_ratings import HumanIrrReport


def _load_summaries(path: Path) -> list[SystemMetricSummary]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("summary file must contain a JSON list")
    return [SystemMetricSummary.model_validate(item) for item in payload]


def _load_comparisons(paths: list[str]) -> list[SystemComparison]:
    comparisons: list[SystemComparison] = []
    for path in paths:
        comparisons.append(
            SystemComparison.model_validate_json(Path(path).read_text(encoding="utf-8"))
        )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description="Render gold experiment Markdown report.")
    parser.add_argument("--summary", required=True, help="system_summary.json")
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--human-irr", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    human_irr = (
        HumanIrrReport.model_validate_json(Path(args.human_irr).read_text(encoding="utf-8"))
        if args.human_irr
        else None
    )
    report = render_gold_experiment_report(
        _load_summaries(Path(args.summary)),
        comparisons=_load_comparisons(args.comparison),
        human_irr=human_irr,
    )
    Path(args.out).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
