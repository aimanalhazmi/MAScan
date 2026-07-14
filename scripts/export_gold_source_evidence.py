"""Export source-anchor evidence checks for the gold-standard dataset."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.exports import csv_text
from mascan.eval.gold_standard import load_gold_standard
from mascan.eval.source_evidence import (
    DEFAULT_ANCHOR_MATCH_THRESHOLD,
    SOURCE_EVIDENCE_CSV_FIELDS,
    render_source_evidence_markdown,
    source_evidence_csv_rows,
    validate_source_anchor_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check source anchors against extracted source PDF text."
    )
    parser.add_argument(
        "--gold-standard",
        default="eval_papers/gold_standard_cases.json",
        help="Path to gold_standard_cases.json.",
    )
    parser.add_argument("--base-dir", default=".")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_ANCHOR_MATCH_THRESHOLD,
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    if args.json_out is None and args.markdown_out is None and args.csv_out is None:
        raise ValueError("Provide --json-out, --markdown-out, --csv-out, or more than one")

    dataset = load_gold_standard(args.gold_standard)
    report = validate_source_anchor_evidence(
        dataset,
        base_dir=args.base_dir,
        match_threshold=args.match_threshold,
    )
    if args.json_out:
        _write_text(args.json_out, report.model_dump_json(indent=2) + "\n")
    if args.markdown_out:
        _write_text(args.markdown_out, render_source_evidence_markdown(report))
    if args.csv_out:
        _write_text(
            args.csv_out,
            csv_text(source_evidence_csv_rows(report), SOURCE_EVIDENCE_CSV_FIELDS),
        )
    return 0 if report.extract_error_count == 0 else 1


def _write_text(path: str, text: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
