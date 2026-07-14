"""Export the strict gold-standard LLM judge rubric as Markdown."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.exports import render_gold_judge_rubric_markdown
from mascan.eval.gold_standard import load_gold_standard


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the gold-standard PESTEL judge system prompt and schema."
    )
    parser.add_argument("--gold-standard", default="eval_papers/gold_standard_cases.json")
    parser.add_argument(
        "--case",
        default=None,
        help="Optional case_id to include a concrete judge user-prompt preview.",
    )
    parser.add_argument(
        "--sample-response-file",
        default=None,
        help="Optional model-response file to include in the case prompt preview.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sample_case = None
    sample_response_text = "[MODEL RESPONSE TEXT GOES HERE]"
    if args.case:
        dataset = load_gold_standard(args.gold_standard)
        sample_case = dataset.by_id(args.case)
    if args.sample_response_file:
        sample_response_text = Path(args.sample_response_file).read_text(encoding="utf-8")

    Path(args.out).write_text(
        render_gold_judge_rubric_markdown(
            sample_case=sample_case,
            sample_response_text=sample_response_text,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
