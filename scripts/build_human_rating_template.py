"""Build a fillable human ratings JSON template from a calibration packet."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.human_calibration import HumanCalibrationPacket
from mascan.eval.human_ratings import build_human_ratings_template


def main() -> int:
    parser = argparse.ArgumentParser(description="Build human ratings template.")
    parser.add_argument("--packet", required=True, help="human_packet.json path.")
    parser.add_argument("--raters", nargs="+", required=True, help="Rater IDs.")
    parser.add_argument("--out", required=True, help="Output template path.")
    args = parser.parse_args()

    packet = HumanCalibrationPacket.model_validate_json(
        Path(args.packet).read_text(encoding="utf-8")
    )
    template = build_human_ratings_template(packet, rater_ids=args.raters)
    Path(args.out).write_text(template.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
