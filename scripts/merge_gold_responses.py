"""Merge multiple response JSON lists into one list for batch judging."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.gold_experiment import ModelResponseRecord


def _load_records(path: Path) -> list[ModelResponseRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [ModelResponseRecord.model_validate(item) for item in payload]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge gold response files.")
    parser.add_argument("--responses", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged: dict[tuple[str, str], ModelResponseRecord] = {}
    for response_path in args.responses:
        for record in _load_records(Path(response_path)):
            key = (record.case_id, record.system_id)
            if key in merged:
                raise ValueError(f"Duplicate response for case/system: {key}")
            merged[key] = record

    records = sorted(merged.values(), key=lambda record: (record.case_id, record.system_id))
    Path(args.out).write_text(
        json.dumps([record.model_dump(mode="json") for record in records], indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
