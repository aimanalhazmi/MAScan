"""Judge one generated PESTEL answer against the 25-case gold-standard dataset.

Usage:
    uv run python scripts/run_gold_judge.py --case 2007_1_SHELL --response-file answer.md
"""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from mascan.core.logging import configure_logging
from mascan.eval.gold_judge import judge_gold_response
from mascan.eval.gold_standard import load_gold_standard


def main() -> int:
    load_dotenv()
    configure_logging()

    parser = argparse.ArgumentParser(description="Run strict PESTEL gold judge.")
    parser.add_argument("--case", required=True, help="Gold-standard case_id.")
    parser.add_argument("--response-file", required=True, help="Model answer file.")
    parser.add_argument(
        "--gold-standard",
        default="eval_papers/gold_standard_cases.json",
        help="Path to gold_standard_cases.json.",
    )
    parser.add_argument("--model", default=None, help="Override judge model.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    dataset = load_gold_standard(args.gold_standard)
    case = dataset.by_id(args.case)
    response_text = Path(args.response_file).read_text(encoding="utf-8")
    result = judge_gold_response(case, response_text, model=args.model)

    payload = result.model_dump(mode="json")
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
