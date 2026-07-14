"""Export a compact reproducibility manifest for the gold-standard dataset."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.exports import (
    GOLD_STANDARD_MANIFEST_CSV_FIELDS,
    csv_text,
    gold_standard_manifest_csv_rows,
    gold_standard_manifest_payload,
    render_gold_standard_manifest_markdown,
)
from mascan.eval.gold_standard import load_gold_standard


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export per-case hashes and coverage counts for the gold standard."
    )
    parser.add_argument(
        "--gold-standard",
        default="eval_papers/gold_standard_cases.json",
        help="Path to gold_standard_cases.json.",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    if args.json_out is None and args.markdown_out is None and args.csv_out is None:
        raise ValueError("Provide --json-out, --markdown-out, --csv-out, or more than one")

    dataset = load_gold_standard(args.gold_standard)
    if args.json_out:
        _write_text(
            args.json_out,
            json.dumps(gold_standard_manifest_payload(dataset), indent=2) + "\n",
        )
    if args.markdown_out:
        _write_text(args.markdown_out, render_gold_standard_manifest_markdown(dataset))
    if args.csv_out:
        _write_text(
            args.csv_out,
            csv_text(
                gold_standard_manifest_csv_rows(dataset),
                GOLD_STANDARD_MANIFEST_CSV_FIELDS,
            ),
        )
    return 0


def _write_text(path: str, text: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
