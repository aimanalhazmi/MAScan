"""Check environment/input prerequisites before running gold experiment."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.preflight import render_gold_preflight_markdown, run_gold_preflight
from mascan.eval.readiness import load_experiment_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Check gold experiment preflight.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--markdown-out",
        default=None,
        help="Optional path for a human-readable Markdown action report.",
    )
    args = parser.parse_args()

    manifest = load_experiment_manifest(args.manifest)
    report = run_gold_preflight(manifest, base_dir=args.base_dir)
    rendered = report.model_dump_json(indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_gold_preflight_markdown(report), encoding="utf-8")
    return 0 if report.is_ready else 1


if __name__ == "__main__":
    sys.exit(main())
