"""Export per-case token, cost, and quality traces from judged records."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import csv
import json
import sys
from pathlib import Path

from mascan.eval.gold_analysis import CaseTraceRecord, case_trace_records
from mascan.eval.gold_experiment import JudgedModelResponse


def _load_judged_records(path: Path) -> list[JudgedModelResponse]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("judged file must contain a JSON list")
    return [JudgedModelResponse.model_validate(item) for item in payload]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export gold case trace rows.")
    parser.add_argument("--judged", required=True, help="Priced judged JSON list.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    if args.json_out is None and args.csv_out is None:
        raise ValueError("Provide --json-out, --csv-out, or both")

    rows = case_trace_records(_load_judged_records(Path(args.judged)))
    payload = [row.model_dump(mode="json") for row in rows]
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(CaseTraceRecord.model_fields),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
