"""Regenerate derived offline artifacts from a gold experiment manifest."""

if __package__:
    from . import _bootstrap  # noqa: F401
else:
    import _bootstrap  # type: ignore  # noqa: F401

import argparse
import sys
from pathlib import Path

from mascan.eval.postprocess import run_gold_postprocess
from mascan.eval.readiness import load_experiment_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess gold experiment artifacts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--readiness-out", default=None)
    args = parser.parse_args()

    manifest = load_experiment_manifest(args.manifest)
    report = run_gold_postprocess(manifest, base_dir=args.base_dir)
    rendered = report.model_dump_json(indent=2)
    if args.readiness_out:
        Path(args.readiness_out).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if report.is_ready else 1


if __name__ == "__main__":
    sys.exit(main())
