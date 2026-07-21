"""Run or print the gold-standard evaluation pipeline."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.pipeline import build_gold_eval_commands, run_pipeline_commands
from mascan.eval.preflight import render_gold_preflight_markdown, run_gold_preflight
from mascan.eval.readiness import load_experiment_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect model outputs, judge them, price/summarize/trace them, "
            "run paired comparisons, and render the evaluation report. "
            "Defaults to dry-run."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--allow-missing-price", action="store_true")
    parser.add_argument(
        "--trace-csv-out",
        default=None,
        help="Optional CSV path for per-case trace rows.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run API/model commands. Omit for dry-run command preview.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --execute, skip commands whose declared outputs already exist.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="With --execute, bypass preflight gating.",
    )
    parser.add_argument(
        "--preflight-out",
        default=None,
        help="Optional path to write the preflight report before execution.",
    )
    parser.add_argument(
        "--preflight-markdown-out",
        default=None,
        help="Optional path to write a human-readable preflight action report.",
    )
    args = parser.parse_args()

    manifest = load_experiment_manifest(args.manifest)
    commands = build_gold_eval_commands(
        manifest,
        judge_model=args.judge_model,
        allow_missing_price=args.allow_missing_price,
        trace_csv_file=args.trace_csv_out,
    )

    if not args.execute:
        print(json.dumps([command.model_dump(mode="json") for command in commands], indent=2))
        return 0

    if not args.skip_preflight:
        report = run_gold_preflight(manifest, base_dir=args.base_dir)
        rendered = report.model_dump_json(indent=2)
        if args.preflight_out:
            out_path = Path(args.preflight_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered + "\n", encoding="utf-8")
        if args.preflight_markdown_out:
            markdown_path = Path(args.preflight_markdown_out)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_gold_preflight_markdown(report), encoding="utf-8")
        if not report.is_ready:
            print(rendered)
            return 1

    run_pipeline_commands(
        commands,
        cwd=args.base_dir,
        skip_existing=args.skip_existing,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
