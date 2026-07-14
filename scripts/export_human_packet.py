"""Export the human calibration packet as Markdown and ratings CSV."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.exports import (
    RATINGS_CSV_FIELDS,
    csv_text,
    ratings_template_csv_rows,
    render_human_packet_markdown,
)
from mascan.eval.human_calibration import HumanCalibrationPacket
from mascan.eval.human_ratings import HumanRatingsTemplate


def main() -> int:
    parser = argparse.ArgumentParser(description="Export human calibration packet.")
    parser.add_argument("--packet", required=True, help="human_packet.json")
    parser.add_argument("--ratings-template", default=None, help="human_ratings_template.json")
    parser.add_argument("--packet-md-out", default=None)
    parser.add_argument("--ratings-csv-out", default=None)
    args = parser.parse_args()

    if args.packet_md_out is None and args.ratings_csv_out is None:
        raise ValueError("Provide --packet-md-out, --ratings-csv-out, or both")
    if args.ratings_csv_out and args.ratings_template is None:
        raise ValueError("--ratings-csv-out requires --ratings-template")

    packet = HumanCalibrationPacket.model_validate_json(
        Path(args.packet).read_text(encoding="utf-8")
    )
    if args.packet_md_out:
        Path(args.packet_md_out).write_text(
            render_human_packet_markdown(packet), encoding="utf-8"
        )
    if args.ratings_csv_out:
        template = HumanRatingsTemplate.model_validate_json(
            Path(args.ratings_template).read_text(encoding="utf-8")
        )
        Path(args.ratings_csv_out).write_text(
            csv_text(ratings_template_csv_rows(template), RATINGS_CSV_FIELDS),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
