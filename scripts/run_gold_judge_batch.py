"""Judge a JSON list of gold-standard response records."""

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
from mascan.eval.gold_experiment import JudgedModelResponse, ModelResponseRecord
from mascan.eval.gold_judge import judge_gold_response
from mascan.eval.gold_standard import load_gold_standard


def _load_records(path: Path) -> list[ModelResponseRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("responses file must contain a JSON list")
    return [ModelResponseRecord.model_validate(item) for item in payload]


def main() -> int:
    load_dotenv()
    configure_logging()

    parser = argparse.ArgumentParser(description="Batch judge gold responses.")
    parser.add_argument("--responses", required=True, help="JSON list of response records.")
    parser.add_argument("--gold-standard", default="eval_papers/gold_standard_cases.json")
    parser.add_argument("--model", default=None, help="Override judge model.")
    parser.add_argument("--out", required=True, help="Output judged JSON path.")
    args = parser.parse_args()

    dataset = load_gold_standard(args.gold_standard)
    records = _load_records(Path(args.responses))
    judged: list[JudgedModelResponse] = []

    for record in records:
        if record.error:
            judged.append(JudgedModelResponse(response=record, error=record.error))
            continue
        try:
            case = dataset.by_id(record.case_id)
            verdict = judge_gold_response(case, record.response_text, model=args.model)
            judged.append(JudgedModelResponse(response=record, judge=verdict))
        except Exception as exc:
            judged.append(JudgedModelResponse(response=record, error=str(exc)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([record.model_dump(mode="json") for record in judged], indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
