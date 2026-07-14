"""Summarize judged gold-standard experiment outputs by system."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.gold_analysis import summarize_systems
from mascan.eval.gold_experiment import JudgedModelResponse


def _load_judged_records(path: Path) -> list[JudgedModelResponse]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("judged file must contain a JSON list")
    return [JudgedModelResponse.model_validate(item) for item in payload]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize gold experiment results.")
    parser.add_argument("--judged", required=True, help="JSON list of judged records.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    records = _load_judged_records(Path(args.judged))
    payload = [summary.model_dump(mode="json") for summary in summarize_systems(records)]
    rendered = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
