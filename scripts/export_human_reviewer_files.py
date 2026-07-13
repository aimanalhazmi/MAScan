"""Export one packet Markdown and one ratings CSV per human rater."""

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
from mascan.eval.human_calibration import HumanCalibrationPacket, filter_packet_for_rater
from mascan.eval.human_ratings import (
    HumanRatingsTemplate,
    filter_human_ratings_template,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-rater human files.")
    parser.add_argument("--packet", required=True, help="human_packet.json")
    parser.add_argument(
        "--ratings-template",
        required=True,
        help="human_ratings_template.json",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--raters",
        nargs="*",
        default=None,
        help="Optional rater IDs. Defaults to all IDs in the template.",
    )
    args = parser.parse_args()

    packet = HumanCalibrationPacket.model_validate_json(
        Path(args.packet).read_text(encoding="utf-8")
    )
    template = HumanRatingsTemplate.model_validate_json(
        Path(args.ratings_template).read_text(encoding="utf-8")
    )
    rater_ids = args.raters or sorted(
        {
            rating.rater_id
            for rating in [*template.depth_ratings, *template.category_ratings]
        }
    )
    if not rater_ids:
        raise ValueError("No rater IDs found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rater_id in rater_ids:
        rater_packet = filter_packet_for_rater(packet, rater_id=rater_id)
        rater_template = filter_human_ratings_template(template, rater_id=rater_id)
        if not rater_template.depth_ratings and not rater_template.category_ratings:
            raise ValueError(f"No ratings rows found for rater_id={rater_id!r}")
        safe_id = _safe_filename(rater_id)
        (out_dir / f"{safe_id}_packet.md").write_text(
            render_human_packet_markdown(rater_packet),
            encoding="utf-8",
        )
        (out_dir / f"{safe_id}_ratings.csv").write_text(
            csv_text(ratings_template_csv_rows(rater_template), RATINGS_CSV_FIELDS),
            encoding="utf-8",
        )
    return 0


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    sys.exit(main())
