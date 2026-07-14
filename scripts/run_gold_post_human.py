"""Run or print the post-human phase of the gold-standard experiment."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

from mascan.eval.pipeline import (
    bind_manifest_path,
    build_post_human_commands,
    run_pipeline_commands,
)
from mascan.eval.preflight import render_gold_preflight_markdown, run_gold_preflight
from mascan.eval.readiness import load_experiment_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import returned human ratings, regenerate final analysis artifacts, "
            "render methodology, and run readiness. Defaults to dry-run."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument(
        "--ratings-csv",
        nargs="*",
        default=None,
        help="Optional filled rater CSV files to import before postprocess.",
    )
    parser.add_argument(
        "--readiness-out",
        default="eval_results/readiness_report.json",
    )
    parser.add_argument(
        "--methodology-out",
        default="eval_results/gold_methodology_appendix.md",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run commands. Omit for dry-run command preview.",
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
    commands = bind_manifest_path(
        build_post_human_commands(
            manifest,
            ratings_csv_files=args.ratings_csv,
            readiness_out=args.readiness_out,
            methodology_out=args.methodology_out,
        ),
        manifest_path=args.manifest,
    )

    if not args.execute:
        print(json.dumps([command.model_dump(mode="json") for command in commands], indent=2))
        return 0

    if not args.skip_preflight:
        report = run_gold_preflight(
            manifest,
            base_dir=args.base_dir,
            phase="post_human",
            ratings_csv_files=args.ratings_csv,
        )
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
