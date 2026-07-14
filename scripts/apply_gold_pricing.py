"""Apply an explicit pricing table to response or judged evaluation JSON."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.costing import (
    PricingTable,
    apply_pricing_to_judged,
    apply_pricing_to_responses,
)
from mascan.eval.gold_experiment import JudgedModelResponse, ModelResponseRecord


def _load_responses(path: Path) -> list[ModelResponseRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("responses file must contain a JSON list")
    return [ModelResponseRecord.model_validate(item) for item in payload]


def _load_judged(path: Path) -> list[JudgedModelResponse]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("judged file must contain a JSON list")
    return [JudgedModelResponse.model_validate(item) for item in payload]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply model pricing to eval JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--responses", help="ModelResponseRecord JSON list.")
    group.add_argument("--judged", help="JudgedModelResponse JSON list.")
    parser.add_argument("--pricing", required=True, help="PricingTable JSON.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-missing-price", action="store_true")
    args = parser.parse_args()

    pricing = PricingTable.model_validate_json(
        Path(args.pricing).read_text(encoding="utf-8")
    )
    require_price = not args.allow_missing_price
    if args.responses:
        records = apply_pricing_to_responses(
            _load_responses(Path(args.responses)),
            pricing,
            require_price=require_price,
        )
        payload = [record.model_dump(mode="json") for record in records]
    else:
        records = apply_pricing_to_judged(
            _load_judged(Path(args.judged)),
            pricing,
            require_price=require_price,
        )
        payload = [record.model_dump(mode="json") for record in records]

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
