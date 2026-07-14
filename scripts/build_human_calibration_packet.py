"""Build a distributed human calibration packet from saved model outputs."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.gold_experiment import ModelResponseRecord
from mascan.eval.gold_standard import load_gold_standard
from mascan.eval.human_calibration import build_human_calibration_bundle


def _load_records(path: Path) -> list[ModelResponseRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("responses file must contain a JSON list")
    return [ModelResponseRecord.model_validate(item) for item in payload]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a distributed human calibration packet. Each rater is assigned "
            "a disjoint subset of cases; together they cover the full case set."
        )
    )
    parser.add_argument("--responses", required=True, help="JSON list of response records.")
    parser.add_argument("--systems", nargs="+", required=True, help="System IDs to include.")
    parser.add_argument("--gold-standard", default="eval_papers/gold_standard_cases.json")
    parser.add_argument(
        "--raters",
        nargs="+",
        required=True,
        help="Rater IDs. Cases are partitioned evenly across these raters.",
    )
    parser.add_argument(
        "--cases-per-rater",
        type=int,
        default=5,
        help="Number of disjoint cases assigned to each rater.",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--packet-out", required=True)
    parser.add_argument("--answer-key-out", required=True)
    args = parser.parse_args()

    dataset = load_gold_standard(args.gold_standard)
    records = _load_records(Path(args.responses))
    bundle = build_human_calibration_bundle(
        dataset,
        records,
        systems=args.systems,
        rater_ids=args.raters,
        cases_per_rater=args.cases_per_rater,
        seed=args.seed,
    )

    Path(args.packet_out).write_text(
        bundle.packet.model_dump_json(indent=2), encoding="utf-8"
    )
    Path(args.answer_key_out).write_text(
        json.dumps(
            [entry.model_dump(mode="json") for entry in bundle.answer_key],
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
