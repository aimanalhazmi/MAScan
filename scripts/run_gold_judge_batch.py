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


def _load_existing_judgments(path: Path) -> dict[str, JudgedModelResponse]:
    """Read judgments already produced so a run can resume without re-paying."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    existing: dict[str, JudgedModelResponse] = {}
    for item in payload if isinstance(payload, list) else []:
        try:
            judged = JudgedModelResponse.model_validate(item)
        except Exception:  # noqa: BLE001 - a partial file must not block the rerun
            continue
        existing[judged.response.case_id] = judged
    return existing


def _write_judgments(path: Path, judged: list[JudgedModelResponse]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([record.model_dump(mode="json") for record in judged], indent=2),
        encoding="utf-8",
    )


def main() -> int:
    load_dotenv()
    configure_logging()

    parser = argparse.ArgumentParser(description="Batch judge gold responses.")
    parser.add_argument("--responses", required=True, help="JSON list of response records.")
    parser.add_argument("--gold-standard", default="eval_papers/gold_standard_cases.json")
    parser.add_argument("--model", default=None, help="Override judge model.")
    parser.add_argument("--out", required=True, help="Output judged JSON path.")
    parser.add_argument(
        "--grounding",
        action="store_true",
        help=(
            "Also run the separate grounding pass (one extra judge call per "
            "response). Reported as a secondary diagnostic; it is not part of "
            "combined quality and does not affect depth or categorization."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse judgments already present in --out and only judge the missing "
            "cases. Judgments are written after every case, so an interrupted run "
            "never has to re-pay for completed judge calls."
        ),
    )
    args = parser.parse_args()

    dataset = load_gold_standard(args.gold_standard)
    records = _load_records(Path(args.responses))
    out_path = Path(args.out)
    existing = _load_existing_judgments(out_path) if args.resume else {}
    judged: list[JudgedModelResponse] = []

    for index, record in enumerate(records, start=1):
        prior = existing.get(record.case_id)
        if prior is not None and prior.judge is not None:
            print(f"[{index}/{len(records)}] {record.case_id}: reusing judgment", flush=True)
            judged.append(prior)
            _write_judgments(out_path, judged)
            continue
        if record.error:
            judged.append(JudgedModelResponse(response=record, error=record.error))
            _write_judgments(out_path, judged)
            continue
        print(f"[{index}/{len(records)}] {record.case_id}: judging...", flush=True)
        try:
            case = dataset.by_id(record.case_id)
            verdict = judge_gold_response(
                case,
                record.response_text,
                model=args.model,
                include_grounding=args.grounding,
            )
            judged.append(JudgedModelResponse(response=record, judge=verdict))
        except Exception as exc:
            print(f"    -> FAILED: {type(exc).__name__}: {exc}", flush=True)
            judged.append(JudgedModelResponse(response=record, error=str(exc)))

        # Persist after every case so an interruption never discards paid calls.
        _write_judgments(out_path, judged)

    failed = sum(1 for record in judged if record.judge is None)
    print(f"\nJudged {len(judged) - failed}/{len(judged)} cases ({failed} failed).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
