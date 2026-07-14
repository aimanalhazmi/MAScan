"""Compute IRR between human raters and the LLM judge."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.gold_experiment import JudgedModelResponse
from mascan.eval.human_calibration import HumanAnswerKeyEntry, HumanCalibrationPacket
from mascan.eval.human_ratings import (
    HumanRatingsFile,
    compute_human_irr_report,
    validate_complete_human_ratings,
)


def _load_judged(path: Path) -> list[JudgedModelResponse]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("judged file must contain a JSON list")
    return [JudgedModelResponse.model_validate(item) for item in payload]


def _load_answer_key(path: Path) -> list[HumanAnswerKeyEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("answer key file must contain a JSON list")
    return [HumanAnswerKeyEntry.model_validate(item) for item in payload]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute human calibration IRR.")
    parser.add_argument("--ratings", required=True, help="HumanRatingsFile JSON.")
    parser.add_argument("--judged", required=True, help="JudgedModelResponse JSON list.")
    parser.add_argument("--answer-key", required=True, help="Coordinator answer key JSON.")
    parser.add_argument("--packet", default=None, help="Optional human_packet.json for completeness validation.")
    parser.add_argument("--raters", nargs="*", default=None, help="Expected rater IDs for completeness validation.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Compute IRR even if completeness validation fails.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ratings = HumanRatingsFile.model_validate_json(
        Path(args.ratings).read_text(encoding="utf-8")
    )
    if args.packet and args.raters:
        packet = HumanCalibrationPacket.model_validate_json(
            Path(args.packet).read_text(encoding="utf-8")
        )
        validation = validate_complete_human_ratings(
            ratings, packet, rater_ids=args.raters
        )
        if not validation.is_complete and not args.allow_incomplete:
            raise ValueError(
                "Human ratings are incomplete or inconsistent: "
                f"{validation.model_dump(mode='json')}"
            )
    report = compute_human_irr_report(
        ratings,
        _load_judged(Path(args.judged)),
        _load_answer_key(Path(args.answer_key)),
    )
    rendered = report.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
