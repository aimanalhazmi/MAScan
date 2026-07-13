"""Run paired significance tests between two judged systems."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.gold_analysis import compare_systems
from mascan.eval.gold_experiment import JudgedModelResponse


def _load_judged_records(path: Path) -> list[JudgedModelResponse]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("judged file must contain a JSON list")
    return [JudgedModelResponse.model_validate(item) for item in payload]


def _assume_normal(value: str) -> bool | None:
    if value == "auto":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("normality must be auto, true, or false")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two gold experiment systems.")
    parser.add_argument("--judged", required=True, help="JSON list of judged records.")
    parser.add_argument("--treatment-system", required=True)
    parser.add_argument("--control-system", required=True)
    parser.add_argument(
        "--metric",
        choices=["analytical_depth", "categorization_accuracy", "combined_quality"],
        default="combined_quality",
    )
    parser.add_argument(
        "--normality",
        type=_assume_normal,
        default=None,
        help="auto, true, or false. Default auto uses Shapiro-Wilk if SciPy exists.",
    )
    parser.add_argument(
        "--normality-alpha",
        type=float,
        default=0.05,
        help="Alpha threshold for Shapiro-Wilk normality selection.",
    )
    parser.add_argument(
        "--alternative",
        choices=["two-sided", "greater", "less"],
        default="two-sided",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    records = _load_judged_records(Path(args.judged))
    comparison = compare_systems(
        records,
        treatment_system=args.treatment_system,
        control_system=args.control_system,
        metric=args.metric,
        assume_normal=args.normality,
        normality_alpha=args.normality_alpha,
        alternative=args.alternative,
    )
    rendered = comparison.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
