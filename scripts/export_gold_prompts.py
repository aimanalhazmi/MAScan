"""Export the 25-case prompt pack as Markdown and/or CSV."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.exports import (
    PROMPT_PACK_CSV_FIELDS,
    csv_text,
    prompt_pack_csv_rows,
    render_prompt_pack_markdown,
)
from mascan.eval.gold_standard import load_gold_standard


def main() -> int:
    parser = argparse.ArgumentParser(description="Export gold-standard prompts.")
    parser.add_argument("--gold-standard", default="eval_papers/gold_standard_cases.json")
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    if args.markdown_out is None and args.csv_out is None:
        raise ValueError("Provide --markdown-out, --csv-out, or both")

    dataset = load_gold_standard(args.gold_standard)
    if args.markdown_out:
        Path(args.markdown_out).write_text(
            render_prompt_pack_markdown(dataset), encoding="utf-8"
        )
    if args.csv_out:
        Path(args.csv_out).write_text(
            csv_text(prompt_pack_csv_rows(dataset), PROMPT_PACK_CSV_FIELDS),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
