"""Export the case-by-case gold-standard validation report as Markdown."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.exports import render_gold_standard_validation_report
from mascan.eval.gold_standard import load_gold_standard


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export prompt, expected answer, and reread notes per gold case."
    )
    parser.add_argument("--gold-standard", default="eval_papers/gold_standard_cases.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = load_gold_standard(args.gold_standard)
    Path(args.out).write_text(
        render_gold_standard_validation_report(dataset), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
